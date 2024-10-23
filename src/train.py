import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import copy
import torchvision.models as models
from dataloaders import *
import argparse
from torch.nn import DataParallel

"""
Example terminal command:

python src/train.py --dataset waterbirds --method base --checkpoint_path /media/exx/HDD/rwiddhic/checkpoints/<dataset_name>_<method_name>.pth or <dataset_name>_<method_name>_augmented.pth if augmentation is True

"""
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        """
        Args:
            patience (int): How many epochs to wait after last time validation loss improved.
                            Default: 7
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                               Default: 0
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss > self.best_score - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0

def train_and_validate(model, train_loader, val_loader, criterion, scheduler, optimizer, num_epochs, device, checkpoint_path):
    
    #model = DataParallel(model).to(device)
    model.to(device)
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    early_stopper = EarlyStopping(patience=10, min_delta=0.001)

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = 100 * correct / total
        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%')

        scheduler.step()
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_loss = val_loss / total
        val_acc = 100 * correct / total
        print(f'Epoch {epoch + 1}/{num_epochs}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')


        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            save_checkpoint({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename=checkpoint_path)


    # Call early stopping logic
        early_stopper(val_loss)
        if early_stopper.early_stop:
            print("Early stopping")
            break
        
    print('Finished Training. Best Validation Accuracy: {:.2f}%'.format(best_acc))
    model.load_state_dict(best_model_wts)
    return model

def save_checkpoint(state, filename='checkpoint.pth'):
    """
    Saves the model and optimizer state to a file.
    
    Args:
        state (dict): State dictionary containing model and optimizer states.
        filename (str): Path to the file where to save the checkpoint.
    """
    print('Saving checkpoint')
    torch.save(state, filename)


parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument('--checkpoint_path', type=str, default='checkpoint.pth.tar', help='Path to save the model checkpoint')
parser.add_argument('--dataset', type=str, default='waterbirds', help='Dataset to train the model on')
parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train the model')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for training')
parser.add_argument('--augmentation', action='store_true', help='Use data augmentation')
parser.add_argument('--method', type=str, default='base', help='Method to train the model', choices=['base', 'conbias', 'alia', 'randaug', 'cutmix'])
parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')

args = parser.parse_args()

print(args)


if torch.cuda.is_available():
    print("Available CUDA devices:")
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No CUDA devices are available.")



# Set device and prepare model, loaders, etc.
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)  # T_max is the number of steps until lr is reset


if args.method == 'base':
    args.augmentation = False

if args.dataset == 'waterbirds':
    train_loader, val_loader, _ = load_waterbirds(augmentation=args.augmentation, method=args.method)

elif args.dataset == 'urbancars':
    train_loader, val_loader, _ = load_urbancars(augmentation=args.augmentation, method=args.method)

elif args.dataset == 'coco-gb':
    train_loader, val_loader, _ = load_cocogb(augmentation=args.augmentation, method=args.method)


for i in range(3):
    print("DATASET: ", args.dataset)
    print("AUGMENTATION: ", args.augmentation)
    print('METHOD: ', args.method)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)  # T_max is the number of steps until lr is reset

    checkpoint_path_new = args.checkpoint_path+f'_run_{i}.pth'
    trained_model = train_and_validate(model, train_loader, val_loader, criterion, scheduler, optimizer, args.num_epochs, device, checkpoint_path_new)

    #print("Training complete. Model saved to", args.checkpoint_path)
    #print("Test Accuracy (Balanced): ", evaluate(trained_model, test_loader, device))
    #print("Test Accuracy (OOD): ", evaluate_ood(trained_model, test_loader, device))