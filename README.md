# WeedGeneration
Channel Attention GAN-based Synthetic Weed Generation for Precise Weed Identification
## Dataset
Please fill [this form](https://docs.google.com/forms/d/e/1FAIpQLSfXBAvp1yO8s7eT1HNVB3BIQDkK4rQ3W5R20J3upR1QGl5zDg/viewform?usp=sf_link) to get download link of the datasets and pretrained weights
## Train
To train the gan model by yourself, please: 
1. Download the ```'dataset.zip'``` following the above ```Dataset``` section and unzip the ```"datasets/"``` to the root of this repo.
2. Check the configuration.py
3. Train the model by running:
    ```
    python train.py
    ```
## Test
To generate synthetic dataset, simply check the arguments in test.py and run:
```
python test.py
```