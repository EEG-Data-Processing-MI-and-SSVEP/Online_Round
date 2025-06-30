import argparse, torch, os, sys, argparse
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

# Get the directory of the current file
current_file_dir = os.path.dirname(__file__)
# Get the parent directory of the current file
parent_dir = os.path.abspath(os.path.join(current_file_dir, '..'))
# Add the parent directory to Python path
sys.path.append(parent_dir)

from data_and_model_classes import SSVEP_dataset, MI_dataset, SSVEP_model_arch, MI_model_arch
from data_and_model_classes.training_related_functions import train_model, plot_training_history



def validate_file_path(path):
    """Validate that the path exists and is a file"""
    path = Path(path)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"File does not exist: {path}")
    return str(path.resolve())  # Return absolute path

def main(args):
    print(f"Training configuration:")
    task = args.task
    train_csv_path = args.train_csv
    val_csv_path = args.validation_csv
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    print(f"Task type: {task}")
    print(f"Train CSV: {train_csv_path}")
    print(f"Validation CSV: {val_csv_path}")
    print(f"Number Of Epochs: {num_epochs}")
    print(f"Batch Size: {batch_size}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    # Intializing Datasets and Loaders, model
    if task.lower() == "mi":
        print("Running Motor Imagery training")
        train_dataset = MI_dataset.MIDataset(train_csv_path, task='MI')
        val_dataset = MI_dataset.MIDataset(train_csv_path, task='MI')
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        model = MI_model_arch.MIPipeline(num_classes=2).to(device)

    elif task.lower() == "ssvep":
        print("Running SSVEP training")
        # Initialize datasets
        train_dataset = SSVEP_dataset.SSVEPDataset(train_csv_path, task='SSVEP')
        val_dataset = SSVEP_dataset.SSVEPDataset(train_csv_path, task='SSVEP')
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        model = SSVEP_model_arch.EnhancedSSVEPModel(num_classes=4).to(device)

    else:
        raise ValueError("Invalid task type. Must be 'MI' or 'SSVEP'")    

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    history, best_model_path = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=num_epochs)
    plot_training_history(history)
    print(f"Best Model Saved at: {best_model_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EEG Training Script')
    # Task type argument
    parser.add_argument(
        '--task',
        type=str,
        required=True,
        choices=['MI', 'SSVEP'],
        help='Task type: MI (Motor Imagery) or SSVEP'
    )
    # Num Epochs
    parser.add_argument(
        '--num_epochs',
        type=int,
        required=True,
        help='Number of Epochs needed for training'
    )
    # Batch Size
    parser.add_argument(
        '--batch_size',
        type=int,
        required=True,
        help='Batch Size needed for training dataloader'
    )
    # CSV file paths
    parser.add_argument(
        '--train_csv',
        type=validate_file_path,
        required=True,
        help='Path to training CSV file'
    )
    parser.add_argument(
        '--validation_csv',
        type=validate_file_path,
        required=True,
        help='Path to validation CSV file'
    )
    args = parser.parse_args()
    main(args)
