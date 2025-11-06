import numpy as np
import pandas as pd
import string
import os

# --- PyTorch Imports ---
# Make sure you have torch installed: pip install torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# --- Scikit-learn Imports ---
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# --- Configuration ---
# File paths from your original script
# ❗ UPDATE THESE PATHS to point to your local files
REAL_DATA_FILE = "Passenger Data A-S.xlsx"
FAKE_DATA_FILE = "fake_names.xlsx"
NAME_COMPONENTS = ['FirstName', 'MiddleName', 'LastName']
RANDOM_STATE = 42

# --- Model Hyperparameters ---
EMBEDDING_DIM = 64   # Size of the vector for each character
LSTM_UNITS = 128     # Number of units in the LSTM layer
DROPOUT_RATE = 0.3
EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.001

# --- Custom Tokenizer and Padding ---

class CharTokenizer:
    """A simple character-level tokenizer."""
    def __init__(self, alphabet=string.ascii_letters + " ", oov_token="<UNK>"):
        self.oov_token = oov_token
        
        # 0 is reserved for padding
        self.char_to_idx = {char: i+1 for i, char in enumerate(sorted(list(alphabet)))}
        self.char_to_idx[oov_token] = len(self.char_to_idx) + 1 # OOV token
        self.vocab_size = len(self.char_to_idx)
        
        # Add padding index
        self.pad_token_idx = 0
        self.oov_token_idx = self.char_to_idx[oov_token]

    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            seq = [self.char_to_idx.get(char, self.oov_token_idx) for char in text]
            sequences.append(seq)
        return sequences

def pad_sequences(sequences, maxlen, padding='post', value=0):
    """Pads sequences to the same length."""
    padded_sequences = np.full((len(sequences), maxlen), fill_value=value, dtype=np.int64)
    for i, seq in enumerate(sequences):
        if len(seq) == 0:
            continue
        if padding == 'post':
            padded_sequences[i, :len(seq)] = seq[:maxlen]
        else: # pre
            padded_sequences[i, -len(seq):] = seq[:maxlen]
    return padded_sequences

# --- PyTorch Dataset Class ---

class NameDataset(Dataset):
    """Custom Dataset for PyTorch."""
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

# --- PyTorch Model Definition ---

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, lstm_units, dropout_rate):
        super(LSTMClassifier, self).__init__()
        
        # Embedding layer
        # vocab_size + 1 because 0 is reserved for padding
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size + 1,
            embedding_dim=embedding_dim,
            padding_idx=0  # Tell the layer that index 0 is padding
        )
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_units,
            num_layers=1,
            batch_first=True,  # Input shape is (batch_size, seq_len, features)
            dropout=dropout_rate if dropout_rate > 0 else 0
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Fully connected layers
        self.fc1 = nn.Linear(lstm_units, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1) # Output 1 logit for binary classification

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        
        # 1. Embedding
        embedded = self.embedding(x)
        # embedded shape: (batch_size, seq_len, embedding_dim)
        
        # 2. LSTM
        # We only care about the output of the last time step
        # lstm_out shape: (batch_size, seq_len, lstm_units)
        # hidden[0] shape: (num_layers, batch_size, lstm_units)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # We'll use the last hidden state
        # Squeeze to remove the 'num_layers' dimension
        last_hidden_state = hidden.squeeze(0)
        # last_hidden_state shape: (batch_size, lstm_units)
        
        # 3. Fully Connected Layers
        out = self.relu(self.fc1(last_hidden_state))
        out = self.dropout(out)
        
        # 4. Output Layer
        out = self.fc2(out)
        # out shape: (batch_size, 1) - these are raw logits
        
        return out

# --- Helper Functions ---

def check_gpu():
    """Checks for GPU and sets the device."""
    print("--- 1. GPU Check ---")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # --- FIX: Removed the checkmark emoji ---
        print(f"GPU is available: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        # --- FIX: Removed the warning emoji ---
        print("No GPU detected. Model will train on CPU.")
    print("-" * 40 + "\n")
    return device

def load_all_names(filepath):
    """
    Loads names from an Excel file, combining FirstName, MiddleName, and LastName.
    (This function is identical to the one in the Keras script)
    """
    try:
        df = pd.concat(pd.read_excel(filepath, sheet_name=None), ignore_index=True)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}. Please check the path.")
        return []
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return []

    for col in NAME_COMPONENTS:
        if col not in df.columns:
            df[col] = ''
        else:
            df[col] = df[col].astype(str).fillna('').str.strip()

    full_names = df[NAME_COMPONENTS[0]]
    for col in NAME_COMPONENTS[1:]:
        full_names = full_names.str.cat(df[col], sep=' ', na_rep='')
    
    names = full_names.str.replace(r'\s+', ' ', regex=True).str.strip()
    names_list = names[names.str.len() > 0].tolist()
    
    if not names_list:
        print(f"Warning: No valid names found in {filepath} after processing.")
        
    return names_list

# ====================================================================
# --- Main Execution ---
# ====================================================================
if __name__ == "__main__":
    
    # 1. Check for GPU
    device = check_gpu()

    # 2. Load and Combine Data
    print("--- 2. Loading Data ---")
    names_real = load_all_names(REAL_DATA_FILE)
    names_fake = load_all_names(FAKE_DATA_FILE)
    
    if not names_real or not names_fake:
        print("\nError: Could not load one or both datasets. Exiting.")
        print("Please check the file paths at the top of the script.")
        exit()
        
    print(f"Loaded {len(names_real)} real names.")
    print(f"Loaded {len(names_fake)} fake names.")

    all_names = names_real + names_fake
    labels = np.array([0] * len(names_real) + [1] * len(names_fake))
    print(f"Total names: {len(all_names)}\n")

    # 3. Tokenize and Pad Sequences
    print("--- 3. Tokenizing and Padding ---")
    tokenizer = CharTokenizer()
    sequences = tokenizer.texts_to_sequences(all_names)
    
    if not sequences:
        print("Error: No sequences were generated from the data. Exiting.")
        exit()
        
    max_seq_length = max(len(seq) for seq in sequences)
    
    X = pad_sequences(sequences, maxlen=max_seq_length, padding='post', value=tokenizer.pad_token_idx)
    y = labels
    
    vocab_size = tokenizer.vocab_size
    print(f"Vocabulary size (from alphabet): {vocab_size}")
    print(f"Max sequence length (padding): {max_seq_length}\n")

    # 4. Train-Test Split
    print("--- 4. Splitting Data ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}\n")

    # 5. Create PyTorch DataLoaders
    train_dataset = NameDataset(X_train, y_train)
    test_dataset = NameDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 6. Build, Compile Model
    print("--- 5. Building Model ---")
    model = LSTMClassifier(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        lstm_units=LSTM_UNITS,
        dropout_rate=DROPOUT_RATE
    ).to(device) # <-- Move model to GPU
    
    # Loss and Optimizer
    # BCEWithLogitsLoss is more stable than Sigmoid + BCELoss
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(model)
    print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n")

    # 7. Train the Model
    print("--- 6. Starting Model Training ---")
    print("Watch your GPU usage via 'nvidia-smi' in a terminal.")
    
    for epoch in range(EPOCHS):
        model.train() # Set model to training mode
        total_train_loss = 0
        
        for inputs, labels in train_loader:
            # Move data to the device (GPU)
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Reshape labels to (batch_size, 1) for the loss function
            labels = labels.unsqueeze(1)
            
            # --- Forward pass ---
            optimizer.zero_grad() # Clear old gradients
            outputs = model(inputs) # Get logits
            loss = criterion(outputs, labels)
            
            # --- Backward pass ---
            loss.backward() # Calculate gradients
            optimizer.step() # Update weights
            
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)
        
        # --- Validation ---
        model.eval() # Set model to evaluation mode
        total_val_loss = 0
        all_preds = []
        all_true = []
        
        with torch.no_grad(): # Disable gradient calculation
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                labels_unsqueezed = labels.unsqueeze(1)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels_unsqueezed)
                total_val_loss += loss.item()
                
                # Convert logits to probabilities
                probs = torch.sigmoid(outputs)
                # Convert probabilities to 0/1 predictions
                preds = (probs > 0.5).float()
                
                all_preds.extend(preds.cpu().numpy())
                all_true.extend(labels.cpu().numpy())
                
        avg_val_loss = total_val_loss / len(test_loader)
        val_accuracy = accuracy_score(all_true, all_preds)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_accuracy:.4f}")

    print("\nTraining complete.\n")

    # 8. Evaluate on Test Set
    print("--- 7. Evaluating Model ---")
    # We already have the final metrics from the last epoch
    print(f"Test Accuracy: {val_accuracy * 100:.2f}%")
    print(f"Test Loss: {avg_val_loss:.4f}")

    print("\nClassification Report:")
    print(classification_report(all_true, all_preds, target_names=['Real (0)', 'Fake (1)'], zero_division=0))

    # 9. Test on custom samples
    print("\n" + "=" * 60)
    print("Sample Prediction Tests")
    print("=" * 60)
    
    test_names = [
        "John Fitzgerald Kennedy",
        "Bartholomew C",
        "Xzyqlph",
        "aaaaabbbbbcc",
        "Elon Musk",
        "Siti Nurhaliza",
        "Rjklsx Vq"
    ]

    # Preprocess the custom names
    test_seq = tokenizer.texts_to_sequences(test_names)
    test_padded = pad_sequences(test_seq, maxlen=max_seq_length, padding='post', value=tokenizer.pad_token_idx)
    
    # Convert to tensor and move to device
    inputs_tensor = torch.tensor(test_padded, dtype=torch.long).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(inputs_tensor) # Get logits
        probs = torch.sigmoid(outputs) # Get probabilities
    
    # Move back to CPU for numpy/printing
    sample_preds = probs.cpu().numpy().flatten()
    
    for name, prob in zip(test_names, sample_preds):
        classification = "Fake" if prob > 0.5 else "Real"
        confidence_fake = prob * 100
        confidence_real = (1 - prob) * 100
        
        print(f"Test Name: '{name}'")
        print(f"-> Classification: {classification}")
        print(f"-> Confidence (Real/Fake): {confidence_real:.2f}% / {confidence_fake:.2f}%")
        print("-" * 20)