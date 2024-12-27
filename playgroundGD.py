'''
This file is intended to test some interval functions 
of this project
'''
from dataset.datasetGD import UIDataset
def main():

    # read image and prompt from the new dataset
    category_path = "./data/categories.txt"
    input_shape = (1920, 1280) # （W, H） 
    test_dataset = UIDataset(data_path="./data", category_path=category_path, input_shape=input_shape) 
    
    image = test_dataset[0]
    print("shape of the image:", image.shape) # （C, H, W）

    
if __name__ == '__main__':
    main()