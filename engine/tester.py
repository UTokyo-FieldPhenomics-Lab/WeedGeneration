import os
import numpy as np
import pandas as pd
import imageio

from tqdm import tqdm

import torch
import torch.nn as nn

class tester:
    def __init__(self, args, model, device):
        self.args = args
        self.device = device

        if args.dataset == 'isas':
            self.classes = ["AMACH", "ABUTH", "BIDPI", "POLLN", "CHEAL"]
        elif args.dataset == 'spsd':
            self.classes = ['SINAR', 'GALAP', 'STEME', 'CHEAL', 'ZEAMX', 'MATIN', 'CAPBP', 'GERPU', 'BEAVA']
        self.n_classes = len(self.classes) # number of classes

        model_name = args.model
        dataset_name = args.dataset
        i = 0
        self.project_name = f'{args.output_dir}/{model_name}_{dataset_name}_{i}'
        while os.path.exists(self.project_name):
            i += 1
            self.project_name = f'{args.output_dir}/{model_name}_{dataset_name}_{i}'
        os.makedirs(self.project_name)

        self.gen = model.Generator(args)
        # DataParallel for multi-gpu model weight
        self.gen = nn.DataParallel(self.gen).to(self.device)

        self.load_model_weights()
        self.fake_list = []
        self.dict = {}

    def load_model_weights(self):
        checkpoint_dir = self.args.model_weight_path
        checkpoint = torch.load(checkpoint_dir)
        gen_weight = checkpoint["G_state_dict"]
        self.gen.load_state_dict(gen_weight)

    def generation(self):
        self.gen.eval()
        with torch.no_grad():
            print('generating images')
            for i in tqdm(range(self.args.unit_number), total=self.args.unit_number):
                fake_noise = torch.FloatTensor(10*self.n_classes, self.args.z_dim).normal_(0.0, 1.0).to(self.device)
                labels = 10 * [i for i in range(self.n_classes)]
                fake_labels = torch.tensor(labels).to(self.device)
                fake = self.gen(fake_noise, fake_labels)
                fake = (fake + 1) / 2
                fake = fake.detach().cpu().tolist()
                torch.cuda.empty_cache()
                self.fake_list.extend(fake)
            # print(len(fake_list))

    def save(self):
        print('saving images')
        img_ids = []
        img_dirs = []
        class_ids = []
        class_names = []
        for i, img in tqdm(enumerate(self.fake_list), total=len(self.fake_list)):
            img = np.array(img).transpose(1, 2, 0)*255
            name = self.classes[i%self.n_classes]
            img_path = f'{self.project_name}/images/{name}/{name}_{i // self.n_classes}.jpg'
            img_name = f'{name}_{i // self.n_classes}.jpg'

            img_ids.append(img_name)
            img_dirs.append(img_path)
            class_ids.append(i%self.n_classes)
            class_names.append(name)
            
            os.makedirs(f'{self.project_name}/images/{name}', exist_ok=True)
            imageio.imwrite(img_path, img.astype(np.uint8))

        self.dict['img_id'] = img_ids
        self.dict['img_dir'] = img_dirs
        self.dict['class_id'] = class_ids
        self.dict['class_name'] = class_names

        # os.makedirs(f'{self.args.project_name}/annotations/', exist_ok=True)
        # annotation_df = pd.DataFrame(self.dict)
        # annotation_df.to_csv(f'{self.args.project_name}/annotations/annotations.csv', index_label=None)


    # tester.evaluation()
    # tester.evaluation_lpips()