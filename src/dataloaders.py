import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage, autoaugment
from torchvision import datasets

class CustomDataset(Dataset):
    """ Custom dataset that loads images based on a DataFrame containing image paths and labels."""
    def __init__(self, dataframe, root_dir, transform=None):
        """
        Args:
            dataframe (DataFrame): DataFrame containing the image paths and labels.
            root_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform
        self.topil = ToPILImage()

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        #print("INDEX: ", self.dataframe.img_filename[idx])
        img_name = os.path.join(self.root_dir, self.dataframe.iloc[idx]['img_filename']) 
        image = read_image(img_name)
        image = self.topil(image).convert('RGB')
        label = self.dataframe.iloc[idx]['y']
        group = self.dataframe.iloc[idx]['place']
        if self.transform:
            image = self.transform(image)
        return img_name, image, label, group
    
def sample_images(df, k=400):
    # Sample k instances from each class
    sampled_data_0 = df[df['y'] == 0].sample(n=k, random_state=42)
    sampled_data_1 = df[df['y'] == 1].sample(n=k, random_state=42)

    # Concatenate the samples into a single DataFrame
    sampled_data = pd.concat([sampled_data_0, sampled_data_1])

    # Reset the index of the resulting DataFrame
    sampled_data.reset_index(drop=True, inplace=True)

    return sampled_data

def load_waterbirds(data_dir='/media/exx/HDD/rwiddhic/waterbird_complete95_forest2water2', filename='metadata.csv', batch_size=32
                    , num_workers=4, shuffle=True, augmentation=False, method='base', additional_filename='additional_metadata.csv'):
    # Load the metadata.csv file
    metadata_path = os.path.join(data_dir, filename)
    metadata = pd.read_csv(metadata_path)

    # Filter by split
    train_df = metadata[metadata['split'] == 0]
    val_df = metadata[metadata['split'] == 1]
    test_df = metadata[metadata['split'] == 2]

    if augmentation:
        # Load additional data if exists
        if method == 'conbias':
            root_dir = '/media/exx/HDD/rwiddhic/aug_datasets/ConBias/Waterbirds'
        elif method == 'alia':
            root_dir = '/media/exx/HDD/rwiddhic/aug_datasets/ALIA/Waterbirds'

        #IMPLEMENT THIS
        elif method == 'cutmix':
            root_dir = '/media/exx/HDD/rwiddhic/aug_datasets/CutMix/Waterbirds'
        
        additional_data_path = os.path.join(root_dir, additional_filename)
        print("Using path: ", additional_data_path)
        if os.path.exists(additional_data_path):
            additional_data = pd.read_csv(additional_data_path)
            additional_data = sample_images(additional_data, k=400)
            train_df = pd.concat([train_df, additional_data], ignore_index=True)
            print(train_df.shape)

    # Define transformations
    randaug_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        autoaugment.RandAugment(num_ops=2, magnitude=9),  # RandAugment for training
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    raw_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    common_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    if method == 'randaug':
        train_dataset = CustomDataset(train_df, data_dir, transform=randaug_transform)
        val_dataset = CustomDataset(val_df, data_dir, transform=common_transform)
        test_dataset = CustomDataset(test_df, data_dir, transform=common_transform)
    elif method == 'raw':
        train_dataset = CustomDataset(train_df, data_dir, transform=raw_transform)
        val_dataset = CustomDataset(val_df, data_dir, transform=raw_transform)
        test_dataset = CustomDataset(test_df, data_dir, transform=raw_transform)
    else:
        train_dataset = CustomDataset(train_df, data_dir, transform=common_transform)
        val_dataset = CustomDataset(val_df, data_dir, transform=common_transform)
        test_dataset = CustomDataset(test_df, data_dir, transform=common_transform)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

def load_cocogb(data_dir='/media/exx/HDD/rwiddhic/coco', filename='metadata_v1.csv', batch_size=32
                    , num_workers=4, shuffle=True, augmentation=False, method='base', 
                    additional_filename='additional_metadata_coco.csv', num_samples=260):
    # Load the metadata.csv file
    metadata_path = os.path.join(data_dir, filename)
    metadata = pd.read_csv(metadata_path)

    # Filter by split
    train_df = metadata[metadata['split'] == 0]
    val_df = metadata[metadata['split'] == 1]
    test_df = metadata[metadata['split'] == 2]

    if augmentation:
        # Load additional data if exists
        if method == 'conbias':
            root_dir = '/media/exx/HDD/rwiddhic/aug_datasets/ConBias/COCO-GB'
        elif method == 'alia':
            root_dir = '/media/exx/HDD/rwiddhic/aug_datasets/ALIA/COCO-GB'

        #IMPLEMENT THIS
        elif method == 'cutmix':
            root_dir = '/media/exx/HDD/rwiddhic/aug_datasets/CutMix/COCOGB'
        
        additional_data_path = os.path.join(root_dir, additional_filename)
        print("Using path: ", additional_data_path)
        if os.path.exists(additional_data_path):
            additional_data = pd.read_csv(additional_data_path)
            #additional_data = sample_images(additional_data, k=150)
            additional_data = additional_data.sample(n=num_samples, random_state=42)
            train_df = pd.concat([train_df, additional_data], ignore_index=True)

        else:
            print("Additional data path does not exist")
            print("Shape of data: ", train_df.shape, val_df.shape, test_df.shape)

    # Define transformations
    randaug_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        autoaugment.RandAugment(num_ops=2, magnitude=9),  # RandAugment for training
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    common_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    if method == 'randaug':
        train_dataset = CustomDataset(train_df, data_dir, transform=randaug_transform)
        val_dataset = CustomDataset(val_df, data_dir, transform=common_transform)
        test_dataset = CustomDataset(test_df, data_dir, transform=common_transform)
    else:
        train_dataset = CustomDataset(train_df, data_dir, transform=common_transform)
        val_dataset = CustomDataset(val_df, data_dir, transform=common_transform)
        test_dataset = CustomDataset(test_df, data_dir, transform=common_transform)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

def load_urbancars(data_dir='/media/exx/HDD/rwiddhic/urbancars_new', filename='metadata.csv', batch_size=32
                    , num_workers=4, shuffle=True, augmentation=False, method='base', additional_filename='additional_metadata_uc.csv'):
    # Load the metadata.csv file
    metadata_path = os.path.join(data_dir, filename)
    metadata = pd.read_csv(metadata_path)

    # Filter by split
    train_df = metadata[metadata['split'] == 0]
    val_df = metadata[metadata['split'] == 1]
    test_df = metadata[metadata['split'] == 2]

    if augmentation:
        # Load additional data if exists
        if method == 'conbias':
            root_dir = '/media/exx/HDD/rwiddhic/aug_datasets/ConBias/UrbanCars'
        elif method == 'alia':
            root_dir = '/media/exx/HDD/rwiddhic/aug_datasets/ALIA/UrbanCars'

        #IMPLEMENT THIS
        elif method == 'cutmix':
            root_dir = '/media/exx/HDD/rwiddhic/aug_datasets/CutMix/UrbanCars'
        
        additional_data_path = os.path.join(root_dir, additional_filename)
        print("Using path: ", additional_data_path)
        if os.path.exists(additional_data_path):
            additional_data = pd.read_csv(additional_data_path)
            #Sample 1300 images from additional data
            additional_data = sample_images(additional_data, k=620)
            #additional_data = additional_data.sample(n=620, random_state=42)
            train_df = pd.concat([train_df, additional_data], ignore_index=True)

    # Define transformations
    randaug_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        autoaugment.RandAugment(num_ops=2, magnitude=9),  # RandAugment for training
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    raw_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    common_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    if method == 'randaug':
        train_dataset = CustomDataset(train_df, data_dir, transform=randaug_transform)
        val_dataset = CustomDataset(val_df, data_dir, transform=common_transform)
        test_dataset = CustomDataset(test_df, data_dir, transform=common_transform)
    elif method == 'raw':
        train_dataset = CustomDataset(train_df, data_dir, transform=raw_transform)
        val_dataset = CustomDataset(val_df, data_dir, transform=raw_transform)
        test_dataset = CustomDataset(test_df, data_dir, transform=raw_transform)
    else:
        train_dataset = CustomDataset(train_df, data_dir, transform=common_transform)
        val_dataset = CustomDataset(val_df, data_dir, transform=common_transform)
        test_dataset = CustomDataset(test_df, data_dir, transform=common_transform)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader