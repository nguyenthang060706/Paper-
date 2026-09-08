import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from sklearn.metrics import classification_report, confusion_matrix
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE_DIR, 'training', 'tier_0_5_synthetic_sessions_train.jsonl')
EVAL_PATH = os.path.join(BASE_DIR, 'training', 'tier_0_5_synthetic_sessions_eval.jsonl')
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'models', 'artifacts', 'v90_tier_0_5_session_risk.pth')

class SyntheticSessionDataset(Dataset):
    def __init__(self, file_path):
        self.sessions = []
        if not os.path.exists(file_path):
            print(f"[Warning] {file_path} not found.")
            return
            
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.sessions.append(json.loads(line))

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, idx):
        session = self.sessions[idx]
        events = session.get('events', [])
        
        max_depth = float(session.get('max_delegation_depth', 0))
        max_fanout = float(session.get('max_fanout_degree', 0))
        
        seq_features = []
        seen_tools = set()
        resource_counts = {}
        
        for ev in events:
            tool = ev.get('tool_name') or 'none'
            resource = ev.get('target_resource') or 'none'
            
            seen_tools.add(tool)
            resource_counts[resource] = resource_counts.get(resource, 0) + 1
            
            iat = float(ev.get('iat_ms_since_prev_in_session', 0.0))
            payload = float(ev.get('payload_bytes', 0.0))
            risk = float(ev.get('content_risk_score', 0.0))
            
            tool_div = float(len(seen_tools))
            res_freq = float(resource_counts[resource])
            
            iat_norm = min(iat / 10000.0, 10.0)
            payload_norm = min(payload / 10000.0, 10.0)
            depth_norm = max_depth / 10.0
            fanout_norm = max_fanout / 10.0
            tool_div_norm = tool_div / 10.0
            res_freq_norm = res_freq / 10.0
            
            seq_features.append([
                iat_norm, payload_norm, risk, 
                depth_norm, fanout_norm, 
                tool_div_norm, res_freq_norm
            ])
            
        if not seq_features:
            seq_features = [[0.0] * 7]
            
        x = torch.tensor(seq_features, dtype=torch.float32)
        label_str = session.get('label', 'benign').lower()
        y = 1.0 if 'malicious' in label_str or 'red' in label_str else 0.0
        
        return x, y

def synthetic_collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    return padded_sequences, lengths, labels

class Phase9LSTMModel(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=64, output_dim=1, num_layers=2):
        super(Phase9LSTMModel, self).__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers, 
            batch_first=True, bidirectional=False, dropout=0.2 if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, lengths):
        x = self.layer_norm(x)
        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (hn, cn) = self.lstm(packed_x)
        last_hidden = hn[-1] 
        out = self.fc(last_hidden)
        return out

def train_model(train_file_path, eval_file_path, epochs=30, batch_size=64, lr=0.001):
    train_dataset = SyntheticSessionDataset(train_file_path)
    eval_dataset = SyntheticSessionDataset(eval_file_path)
    
    if len(train_dataset) == 0 or len(eval_dataset) == 0:
        print("[Error] Datasets are empty.")
        return None
        
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=synthetic_collate_fn
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=synthetic_collate_fn
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Phase9LSTMModel(input_dim=7, hidden_dim=64, num_layers=2).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    print(f"Bắt đầu huấn luyện Phase 9 (7 Features) trên thiết bị: {device}")
    
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for padded_x, lengths, labels in train_loader:
            padded_x, labels = padded_x.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(padded_x, lengths)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * padded_x.size(0)
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
        train_loss = running_loss / total
        train_acc = correct / total * 100
        
        model.eval()
        val_loss = 0.0
        val_total = 0
        with torch.no_grad():
            for padded_x, lengths, labels in eval_loader:
                padded_x, labels = padded_x.to(device), labels.to(device)
                outputs = model(padded_x, lengths)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * padded_x.size(0)
                val_total += labels.size(0)
                
        val_loss /= val_total
        
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {train_loss:.4f} - Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early Stopping kích hoạt ở epoch {epoch+1}!")
                break
                
    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        
    return model

def evaluate_model(model, eval_file_path, batch_size=64, threshold=0.85):
    if model is None:
        return
        
    eval_dataset = SyntheticSessionDataset(eval_file_path)
    eval_loader = DataLoader(
        eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=synthetic_collate_fn
    )
    
    device = next(model.parameters()).device
    model.eval()
    
    all_preds, all_labels = [], []
    
    print(f"\n--- Đánh giá trên tập Eval ({len(eval_dataset)} sessions) với Decision Threshold = {threshold} ---")
    with torch.no_grad():
        for padded_x, lengths, labels in eval_loader:
            padded_x = padded_x.to(device)
            labels = labels.numpy()
            
            outputs = model(padded_x, lengths)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs >= threshold).astype(int)
            
            all_preds.extend(preds)
            all_labels.extend(labels)
            
    cm = confusion_matrix(all_labels, all_preds)
    if cm.size == 1:
        tn, fp, fn, tp = (cm[0,0], 0, 0, 0) if all_labels[0] == 0 else (0, 0, 0, cm[0,0])
    else:
        tn, fp, fn, tp = cm.ravel()
        
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    print(f"TN: {tn} | FP: {fp} | FN: {fn} | TP: {tp}")
    print(f"False Positive Rate (FPR): {fpr*100:.2f}%")
    
    if fpr < 0.05:
        print("QUALITY GATE PASSED: FPR < 5%!")
    else:
        print("QUALITY GATE FAILED: FPR >= 5%.")

    return cm, fpr

if __name__ == "__main__":
    trained_model = train_model(TRAIN_PATH, EVAL_PATH, epochs=30, batch_size=64, lr=0.001)
    # Evaluate at standard production threshold (0.85)
    evaluate_model(trained_model, EVAL_PATH, threshold=0.85)
