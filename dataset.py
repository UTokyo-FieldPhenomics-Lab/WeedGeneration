import imageio as io
from torch.utils.data import Dataset


class WeedDataset(Dataset):
    '''
    WeedDataset
    '''
    def __init__(self, img_df, args, transforms=None):
        super().__init__()
        self.img_ids = img_df["img_id"].to_list()
        self.records = img_df["class_id"].to_list()
        self.args = args
        self.transforms = transforms
    
    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        label_class = self.records[index]
        # print
        # png_img = io.imread(f'{self.args.img_dir}/{img_id}')
        png_img = io.imread(f'{self.args.img_dir}/{img_id}')
        rgb_img = png_img[..., :3]
        binary_mask = 1.0 * (png_img[..., -1] > 0)

        if self.transforms:
            transformed = self.transforms(image=rgb_img, mask=binary_mask)
            rgb_img = transformed['image']
            binary_mask = transformed['mask']
        
        target = {"mask": binary_mask, "class": label_class}

        return rgb_img, target

if __name__ == "__main__":
    pass
    