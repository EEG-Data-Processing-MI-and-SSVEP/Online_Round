from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, task, num_epochs=50, device='cuda'):
    """Enhanced training loop with comprehensive metrics tracking"""
    
    history = {
        'train_loss': [],
        'train_f1': [],
        'val_loss': [],
        'val_f1': [],
        'lr': []
    }

    best_loss = float('inf')
    best_f1   = 0
    best_model_path = None
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        # Training phase with progress bar
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for eeg, freq, motion, labels in pbar:
            eeg = eeg.to(device)
            freq = freq.to(device)
            motion = motion.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(eeg, freq, motion)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        # Calculate training metrics
        train_loss = running_loss / len(train_loader)
        train_f1 = f1_score(all_labels, all_preds, average='macro')
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for eeg, freq, motion, labels in val_loader:
                eeg = eeg.to(device)
                freq = freq.to(device)
                motion = motion.to(device)
                labels = labels.to(device)
                
                outputs = model(eeg, freq, motion)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_f1 = f1_score(val_labels, val_preds, average='macro')
        
        # Update learning rate
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        new_lr = optimizer.param_groups[0]['lr']
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)
        history['lr'].append(current_lr)
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")
        print(f"  LR: {current_lr:.6f}")
        
        # Save best model
        if (val_f1 > best_f1) or (val_loss < best_loss):
            if (val_f1 > best_f1):
                best_f1   = val_f1
            else:
                best_loss = val_loss
                
            best_model_path = f'mtc_best_weights/{task}_model_epoch{epoch+1}_f1{val_f1:.4f}_loss{val_loss:.4f}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_loss': val_loss,
            }, best_model_path)
            print(f"  New best model saved: F1={best_f1:.4f} - Loss{best_loss:.4f}")
        
        print("-" * 60)
    
    return history, best_model_path

def plot_training_history(history):
    """Plot training and validation metrics"""
    plt.figure(figsize=(12, 4))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # F1 score plot
    plt.subplot(1, 2, 2)
    plt.plot(history['train_f1'], label='Train F1')
    plt.plot(history['val_f1'], label='Val F1')
    plt.title('Training and Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    
    plt.tight_layout()
    plt.show()