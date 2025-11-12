# -*- coding: utf-8 -*-

import torch
from torch.utils.data import DataLoader
from testloader import GTSRB_Test_Loader
from evaluation import evaluate
import os

if __name__ == '__main__':
    torch.manual_seed(118)
    testloader = DataLoader(GTSRB_Test_Loader(), 
                    batch_size=50, 
                    shuffle=True, num_workers=8)
    model = None
    # import your trained model 
    pretrained_dir = "pretrained_model"
    for file in os.listdir(pretrained_dir):
        if file.endswith(".pth"):
            model_path = os.path.join(pretrained_dir, file)
            print(f"Evaluating {model_path}...")
            model = torch.load(model_path, map_location=torch.device('cpu'))
            model.eval()
            testing_accuracy = evaluate(model, testloader)
            print('testing finished, accuracy: {:.3f}'.format(testing_accuracy))
    # between
    testing_accuracy = evaluate(model, testloader)
    print('testing finished, accuracy: {:.3f}'.format(testing_accuracy))