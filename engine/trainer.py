import os
import time
import random

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

import torch.backends.cudnn as cudnn

import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset import WeedDataset

# use diffaugment for image augmentation
from DiffAugment_pytorch import DiffAugment as diff
policy = 'color,translation,cutout'

# fix some random seeds
seed = 1123
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
cudnn.benchmark = False
cudnn.deterministic=True

# trainer
class trainer:
    def __init__(self, args, model, device):

        self.args = args
        self.device = device
        
        model_name = args.model
        dataset_name = args.dataset
        i = 0
        self.project_name = f'{args.output_dir}/{model_name}_{dataset_name}_{i}'
        while os.path.exists(self.project_name):
            i += 1
            self.project_name = f'{args.output_dir}/{model_name}_{dataset_name}_{i}'
        os.makedirs(self.project_name)

        self.gen = model.Generator(args)
        self.crit = model.Discriminator(args)

        # DataParallel for multi-gpu training
        self.gen = nn.DataParallel(self.gen).to(self.device)
        self.crit = nn.DataParallel(self.crit).to(self.device)

        self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=self.args.lr, betas=(self.args.beta_1, self.args.beta_2))
        self.crit_opt = torch.optim.Adam(self.crit.parameters(), lr=self.args.lr, betas=(self.args.beta_1, self.args.beta_2))

        transform = A.Compose(
            [
                A.PadIfNeeded(min_height=470, min_width=470, border_mode=cv2.BORDER_CONSTANT, value=0),
                A.CenterCrop(height=400, width=400),
                A.Resize(128, 128),
                A.ShiftScaleRotate(shift_limit=0., scale_limit=0., rotate_limit=90, p=0.5),
                A.VerticalFlip(p=0.5),              
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ToTensorV2(),
            ]
        )
        df = pd.read_csv(self.args.annotation_dir, index_col=None)
        self.dataset = WeedDataset(df, args=args, transforms=transform)

        self.dataloader_train = DataLoader(
            self.dataset,
            batch_size=self.args.batch_size,
            # num_workers=12,
            shuffle=True,
            drop_last=True,
            pin_memory=True, 
            worker_init_fn=np.random.seed(seed)
        )

        self.iters = 0
        self.critic_losses = []
        self.generator_losses = []
    
    def discriminator_criterion(self, real_pred, fake_pred):
        d_loss_real = torch.nn.ReLU()(1.0 - real_pred).mean()                                              
        d_loss_fake = torch.nn.ReLU()(1.0 + fake_pred).mean()
        return (d_loss_real + d_loss_fake) / 2

    def generator_criterion(self, fake_pred):
        return -fake_pred.mean()

    def train(self, ckpt=None):

        if ckpt != None:
            # load_check_point
            checkpoint_dir = ckpt
            checkpoint = torch.load(checkpoint_dir)
            gen_weight = checkpoint["G_state_dict"]
            crit_weight = checkpoint["D_state_dict"]
            self.gen.load_state_dict(gen_weight)
            self.crit.load_state_dict(crit_weight)
            self.iters = checkpoint["iterations"]
            pre_gloss = checkpoint['loss_generator']
            pre_dloss = checkpoint['loss_critic']

        data_iter = iter(self.dataloader_train)

        # fixed noise for visualization
        fixed_noise = torch.FloatTensor(10*self.args.n_classes, self.args.z_dim).normal_(0.0, 1.0).to(self.device)
        fixed_labels = 10 * [i for i in range(self.args.n_classes)]
        fixed_labels = torch.tensor(fixed_labels).to(self.device)

        time_start = time.time()
        time_temp = time.time()
        for _ in range(self.args.iterations):
            # print(f'epoch: {+1}')
            # Dataloader returns the batches
                    # set to train mode
            self.crit.train()
            self.gen.train()
            
            # Dataloader returns the batches
            try:
                images, targets = next(data_iter)
            except:
                data_iter = iter(self.dataloader_train)
                images, targets = next(data_iter)

            reals = images.to(self.device) # normalize reals
            labels = targets["class"].to(self.device)

            d_running_loss = 0
            g_running_loss = 0

            # -----------------
            #  Train Discriminator
            # -----------------
            
            crit_repeats = self.args.crit_repeat
            for _ in range(crit_repeats):
                self.crit_opt.zero_grad()
                # Get noise corresponding to the current batch_size
                fake_noise = torch.FloatTensor(self.args.batch_size, self.args.z_dim).normal_(0.0, 1.0).to(self.device)
                fakes = self.gen(fake_noise, labels)
                # print(fake.shape)

                if self.args.diffaugment == True:
                    diff_reals = diff(reals, policy=policy)
                    diff_fakes = diff(fakes.detach(), policy=policy)

                    fake_pred = self.crit(diff_fakes, labels)
                    real_pred = self.crit(diff_reals, labels)

                else:
                    fake_pred = self.crit(fakes.detach(), labels)
                    real_pred = self.crit(reals, labels)

                disc_loss = self.discriminator_criterion(real_pred, fake_pred)

                # Keep track of the average critic loss in this batch
                d_running_loss += disc_loss.item()

                # discriminator update
                disc_loss.backward(retain_graph=True)
                self.crit_opt.step()
            d_running_loss /= crit_repeats
            self.critic_losses.append(d_running_loss)

            # ---------------------
            #  Train Generator
            # ---------------------
            self.gen_opt.zero_grad()
            if self.args.diffaugment == True:
                diff_fakes = diff(fakes, policy=policy)
                fake_pred = self.crit(diff_fakes, labels)
            else:
                fake_pred = self.crit(fakes, labels)

            gen_loss =  self.generator_criterion(fake_pred)

            # Keep track of the generator losses
            g_running_loss += gen_loss.item()
            self.generator_losses.append(g_running_loss)

            # Update generator
            gen_loss.backward()
            self.gen_opt.step()
            self.iters += 1

            if self.iters % self.args.visualize_step_bin == 0:
                cost_time = time.time() - time_temp
                time_temp = time.time()

                # loss visualization
                steps = self.args.visualize_step_bin
                gen_mean = sum(self.generator_losses[-steps:]) / steps
                critic_mean = sum(self.critic_losses[-steps:]) / steps
                print(f"iter [{self.iters}/{self.args.iterations}], generator loss: {gen_mean:.3f}, discriminator loss: {critic_mean:.3f}, time_cost: {cost_time:.3f}")

                # fixed noise visualization
                fixed_fake = self.gen(fixed_noise, fixed_labels)
                fixed_fake = (fixed_fake + 1) / 2
                fixed_fake = fixed_fake.detach().cpu()
                fixed_image_grid = make_grid(fixed_fake[:50], 
                nrow=5)
                fixed_fig = plt.figure(figsize=(25,25))
                plt.imshow(fixed_image_grid.permute(1, 2, 0).squeeze())
                plt.xticks([])
                plt.yticks([])
                if not os.path.exists(f'{self.project_name}/fixed_visualization'):
                    os.makedirs(f'{self.project_name}/fixed_visualization')
                plt.savefig(f'{self.project_name}/fixed_visualization/iters_{self.iters}.jpg')
                plt.close()

                
                # samples visualization
                fig = plt.figure(figsize=(20,25))
                ax1 = fig.add_subplot(221)
                ax1.set_xticks([])
                ax1.set_yticks([])
                image_tensor = (fakes + 1) / 2
                image_unflat = image_tensor.detach().cpu()
                image_grid = make_grid(image_unflat[:16], nrow=4)
                ax1.imshow(image_grid.permute(1, 2, 0).squeeze())

                ax2 = fig.add_subplot(222)
                ax2.set_xticks([])
                ax2.set_yticks([])
                if self.args.diffaugment == True:
                    image_tensor = (diff_reals + 1) / 2
                else:
                    image_tensor = (reals + 1) / 2
                image_unflat = image_tensor.detach().cpu()
                image_grid = make_grid(image_unflat[:16], nrow=4)
                ax2.imshow(image_grid.permute(1, 2, 0).squeeze())

                ax3 = fig.add_subplot(212)
                step_bins = 10
                num_examples = (len(self.generator_losses) // step_bins) * step_bins
                ax3.plot(
                    range(10, num_examples + 10, step_bins), 
                    torch.Tensor(self.generator_losses[:num_examples]).view(-1, step_bins).mean(1),
                    label="G loss"
                )
                ax3.plot(
                    range(10, num_examples + 10, step_bins), 
                    torch.Tensor(self.critic_losses[:num_examples]).view(-1, step_bins).mean(1),
                    label="D Loss"
                )
                ax3.legend()

                if not os.path.exists(f'{self.project_name}/visualization'):
                    os.makedirs(f'{self.project_name}/visualization')
                plt.savefig(f'{self.project_name}/visualization/iters_{self.iters}.jpg')
                plt.close()

            ## Save Checkpoints ##
            if self.iters % self.args.checkpoints_step_bin == 0:
                if ckpt != None:
                    lg =  np.concatenate((pre_gloss, np.array(self.generator_losses)), axis=0)
                    lc =  np.concatenate((pre_dloss, np.array(self.critic_losses)), axis=0)
                else:
                    lg = np.array(self.generator_losses)
                    lc = np.array(self.critic_losses)
                checkpoint = {
                        'G_state_dict': self.gen.state_dict(),
                        'D_state_dict': self.crit.state_dict(),
                        'loss_generator': lg,
                        'loss_critic': lc,
                        'iterations': self.iters
                    }
                if not os.path.exists(f'{self.project_name}/checkpoints'):
                    os.makedirs(f'{self.project_name}/checkpoints')

                torch.save(checkpoint, f'{self.project_name}/checkpoints/iters_{self.iters}.tar')  # overwrite if exist

        # finish training
        time_end = time.time()
        s=time.gmtime(time_end - time_start)
        time_str = time.strftime("%H:%M:%S", s)
        print(f'training finished, total time cost: {time_str}')
        if ckpt != None:
            lg =  np.concatenate((pre_gloss, np.array(self.generator_losses)), axis=0)
            lc =  np.concatenate((pre_dloss, np.array(self.critic_losses)), axis=0)
        else:
            lg = np.array(self.generator_losses)
            lc = np.array(self.critic_losses)
        checkpoint = {
                'G_state_dict': self.gen.state_dict(),
                'D_state_dict': self.crit.state_dict(),
                'loss_generator': lg,
                'loss_critic': lc,
                'iterations': self.iters
            }
        if not os.path.exists(f'{self.project_name}/checkpoints'):
            os.makedirs(f'{self.project_name}/checkpoints')

        torch.save(checkpoint, f'{self.project_name}/checkpoints/iters_{self.iters}.tar')  # overwrite if exist

                
if __name__ == "__main__":

    import configuration

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    engine = trainer(configuration, device)
    engine.train(ckpt=None)
