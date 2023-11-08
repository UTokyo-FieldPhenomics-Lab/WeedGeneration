### basic configurations ###
model = 'segan' # select the model to use. (["segan", "sngan", "wacgan_info"])
dataset = 'spsd' # select the dataset. (["spsd", "isas"])
output_dir = './results/'
### training configurations ###
diffaugment = False # choose whether to use DiffAugment

visualize_step_bin = 1 # visualization step bin
checkpoints_step_bin = 10000 # step bin to save checkpoint
iterations = 80000 # total iterations to train
z_dim =128 # input noise dimension
batch_size = 50 # image batch size
lr = 0.0002 # initial learning rate
beta_1 = 0.5 # optimizer adam beta_1
beta_2 = 0.9 # optimizer adam beta_2
crit_repeat = 2 # crit repeat number

### model configurations ### 
channels = 3 # image channel number
bottom_width = 4 # initial resolution of generator
gf_dim = 1024 # generator dimension
df_dim = 64 # discriminator dimension
g_spectral_norm = True # whether to use spectral normalization in generator
d_spectral_norm = True # whether to use spectral normalization in discriminator

### dataset configurations ###
img_dir = f'./datasets/images/{dataset}' # dataset image directory
annotation_dir = f'./datasets/annotations/{dataset}/train.csv' # dataset annotation directory
pad_size = 470 # padding size for image preprocessing
img_size = 128 # input image size

if dataset == 'isas':
    classes = ["AMACH", "ABUTH", "BIDPI", "POLLN", "CHEAL"]
elif dataset == 'spsd':
    classes = ['SINAR', 'GALAP', 'STEME', 'CHEAL', 'ZEAMX', 'MATIN', 'CAPBP', 'GERPU', 'BEAVA']
n_classes = len(classes) # number of classes