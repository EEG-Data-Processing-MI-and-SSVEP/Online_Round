import argparse, torch, os, sys, argparse
from torch.utils.data import DataLoader
from dotenv import load_dotenv
import pandas as pd
from pathlib import Path

# Get the directory of the current file
current_file_dir = os.path.dirname(__file__)
# Get the parent directory of the current file
parent_dir = os.path.abspath(os.path.join(current_file_dir, '..'))
# Add the parent directory to Python path
sys.path.append(parent_dir)

from data_and_model_classes import SSVEP_dataset, MI_dataset, SSVEP_model_arch, MI_model_arch


def validate_file_path(path):
    """Validate that the path exists and is a file"""
    path = Path(path)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"File does not exist: {path}")
    return str(path.resolve())  # Return absolute path

def create_submission(test_csv_path, mi_model, ssvep_model):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")
    
    # Load test data separately for MI and SSVEP
    test_mi_dataset = MI_dataset.MIDataset(test_csv_path, task='MI')
    test_ssvep_dataset = SSVEP_dataset.SSVEPDataset(test_csv_path, task='SSVEP')
    
    test_mi_loader = DataLoader(test_mi_dataset, batch_size=16, shuffle=False)
    test_ssvep_loader = DataLoader(test_ssvep_dataset, batch_size=16, shuffle=False)
    
    mi_inverse_label_encoding = {
        0: 'Left',
        1: 'Right'
    }

    ssvep_inverse_label_encoding = {
        2: 'Left',
        3: 'Right',
        1: 'Forward',
        0: 'Backward'
    }
    
    # Dictionary to store predictions {id: label}
    predictions = {}
    
    # Process MI test data
    with torch.no_grad():
        for eeg, motion, _, ids in test_mi_loader:
            eeg    = eeg.permute(0, 2, 1).to(device)
            motion = motion.permute(0, 2, 1).to(device)
            outputs = mi_model(eeg, motion)
            preds = torch.argmax(outputs, dim=1)
            
            for id_num, pred in zip(ids, preds):
                predictions[id_num.item()] = mi_inverse_label_encoding[pred.item()]
    
    # Process SSVEP test data
    with torch.no_grad():
        for eeg, freq, motion, _, ids in test_ssvep_loader:
            eeg = eeg.to(device)
            freq = freq.to(device)
            motion = motion.to(device)
            
            logits = ssvep_model(eeg, freq, motion)
            preds = torch.argmax(logits, dim=1)
            
            for id_num, pred in zip(ids, preds):
                predictions[id_num.item()] = ssvep_inverse_label_encoding[pred.item()]
    
    # Create submission dataframe
    load_dotenv()
    submission_df = pd.read_csv(f"{os.getenv('DATA_BASE_DIR')}/sample_submission.csv")
    
    # Fill in predictions in order of submission file
    for idx in submission_df['id']:
        submission_df.loc[submission_df['id'] == idx, 'label'] = predictions[idx]

    
    # Save submission
    created_submission_path = f"created_submissions/submission.csv"
    submission_df.to_csv(created_submission_path, index=False)
    print(f"Submission file created: submission.csv at {created_submission_path}")
    print(f"Total predictions made: {len(predictions)} (should be 100)")
    
    # Verify we have predictions for all test samples
    test_df = pd.read_csv(f"{os.getenv('DATA_BASE_DIR')}/test.csv")
    missing = set(test_df['id']) - set(predictions.keys())
    if missing:
        print(f"Warning: Missing predictions for IDs: {sorted(missing)}")
    else:
        print("Success: Predictions generated for all test samples")


def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    # Load models
    mi_model = MI_model_arch.MIPipeline(eeg_channels=2, motion_channels=6, num_classes=2).to(device)
    ssvep_model = SSVEP_model_arch.EnhancedSSVEPModel(num_classes=4).to(device)

    # Load weights
    mi_checkpoint = torch.load(args.mi_checkpoint_path, map_location=device, weights_only=False)
    mi_model.load_state_dict(mi_checkpoint['model_state_dict'])

    ssvep_checkpoint = torch.load(args.ssvep_checkpoint_path, map_location=device, weights_only=False)
    ssvep_model.load_state_dict(ssvep_checkpoint['model_state_dict'])

    # Set models to evaluation mode
    mi_model.eval()
    ssvep_model.eval();

    create_submission(args.test_set_csv_path, mi_model, ssvep_model)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EEG Training Script')

    # Checkpoint_pathes
    parser.add_argument(
        '--mi_checkpoint_path',
        type=validate_file_path,
        required=True,
        help='Path to .pth file of best mi_model weights'
    )
    parser.add_argument(
        '--ssvep_checkpoint_path',
        type=validate_file_path,
        required=True,
        help='Path to .pth file of best ssvep_model weights'
    )
    parser.add_argument(
        '--test_set_csv_path',
        type=validate_file_path,
        required=True,
        help='Path to .csv file of testset'
    )
    args = parser.parse_args()
    main(args)