import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import copy
import torchvision.models as models
from dataloaders import *
import argparse
import numpy as np

"""
Example terminal command:

python src/evaluate.py --dataset waterbirds --checkpoints_path /media/exx/HDD/rwiddhic/checkpoints/wb_base.pth
"""
def evaluate(model, test_loader, device):
    model.to(device)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels, _ in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    acc = 100 * correct / total

    return acc
    print(f'Test Accuracy: {acc:.2f}%')

def evaluate_ood(model, test_loader, device, datname='waterbirds'):
    model.to(device)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels, groups in test_loader:
            # Filtering based on the specified conditions
            if datname == 'urbancars':
                mask = (labels == 1) & (groups == 7) | (labels == 0) & (groups == 3)
                filtered_images = images[mask]
                filtered_labels = labels[mask]
                filtered_images, filtered_labels = filtered_images.to(device), filtered_labels.to(device)
                outputs = model(filtered_images)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == filtered_labels).sum().item()
                total += filtered_labels.size(0)
                
            elif datname == 'waterbirds':
                mask = (labels == 1) & (groups == 0) | (labels == 0) & (groups == 1)
                filtered_images = images[mask]
                filtered_labels = labels[mask]
            
                filtered_images, filtered_labels = filtered_images.to(device), filtered_labels.to(device)
                outputs = model(filtered_images)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == filtered_labels).sum().item()
                total += filtered_labels.size(0)

            else:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

    # Avoid division by zero if total is zero
    if total > 0:
        acc = 100 * correct / total
        #print(f'Test Accuracy: {acc:.2f}%')
    else:
        print("No data points met the filtering criteria.")

    return acc

def load_checkpoint(model, filename):
    """
    Loads the model and optimizer state from a file.
    
    Args:
        state (dict): State dictionary containing model and optimizer states.
        filename (str): Path to the file where to load the checkpoint.
    """
    print('Loading checkpoint')

    checkpoint = torch.load(filename)
    
    print(checkpoint.keys())
    try:
        model.load_state_dict(checkpoint['state_dict'])
    except:
        adjusted_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
        model.load_state_dict(adjusted_state_dict)
    return model

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', type=str, default='checkpoint.pth.tar', help='Path to the checkpoint file')
parser.add_argument('--dataset', type=str, default='waterbirds', help='Dataset to use')
parser.add_argument('--type', type=str, default='balanced', help='Evaluation Type', choices=['balanced', 'ood'])
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

if args.dataset == 'waterbirds':
    _, _, test_loader = load_waterbirds()
elif args.dataset == 'coco-gb':
    if args.type == 'ood':
        filename = 'metadata_v2.csv'
    else:
        filename = 'metadata_v1.csv'
    _, _, test_loader = load_cocogb(filename=filename)

elif args.dataset == 'urbancars':
    _, _, test_loader = load_urbancars()

accs = []
if args.type == 'balanced':
    for i in range(3):
        checkpoint_path_new = args.checkpoint_path+f'_run_{i}.pth'
        trained_model = load_checkpoint(model, checkpoint_path_new)
        accs.append(evaluate(trained_model, test_loader, device))
        
    print(f'Test Accuracy (Balanced): {np.mean(accs):.2f}%'
          f' ± {np.std(accs):.2f}%')

else:
    for i in range(3):
        checkpoint_path_new = args.checkpoint_path+f'_run_{i}.pth'
        trained_model = load_checkpoint(model, checkpoint_path_new)
        accs.append(evaluate_ood(trained_model, test_loader, device, args.dataset))
    print(f'Test Accuracy (OOD): {np.mean(accs):.2f}%'
          f' ± {np.std(accs):.2f}%')