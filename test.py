# -*- coding: utf-8 -*-

import torch
from torch.utils.data import DataLoader
from testloader import GTSRB_Test_Loader
from evaluation import evaluate
import os
from predict import load_model
if __name__ == '__main__':
    torch.manual_seed(118)
    testloader = DataLoader(GTSRB_Test_Loader(), 
                    batch_size=50, 
                    shuffle=True, num_workers=8)
    model = None
    # import your trained model 
    pretrained_dir = "trained_models"
    for file in os.listdir(pretrained_dir):
        if file.endswith(".pth"):
            model_path = os.path.join(pretrained_dir, file)
            print(f"Evaluating {model_path}...") 
            # Load the checkpoint
            model = load_model(model_path, device='cpu')[0]
            model.eval()
            testing_accuracy = evaluate(model, testloader)
            print('testing finished, accuracy: {:.3f}'.format(testing_accuracy))
    # between
    if model is None:
        print("No .pth files found in pretrained_model directory")
    testing_accuracy = evaluate(model, testloader)
    print('testing finished, accuracy: {:.3f}'.format(testing_accuracy))