import torch
import torch.nn as nn
import torch.optim as optim
import struct
import os

# ==========================================
# 1. Define the PyTorch Model
# ==========================================
class AgnosMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AgnosMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# ==========================================
# 2. Train the Model (Using synthetic data for demo)
# ==========================================
def train_model():
    print("🚀 Starting training phase...")
    input_size = 20
    hidden_size = 64
    output_size = 2
    
    model = AgnosMLP(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Generate some dummy classification data
    X = torch.randn(1000, input_size)
    y = torch.randint(0, output_size, (1000,))

    # Quick training loop
    model.train()
    for epoch in range(50):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/50], Loss: {loss.item():.4f}")
            
    print("✅ Training complete.")
    return model

# ==========================================
# 3. Export to Custom .agnos Binary Format
# ==========================================
def export_to_agnos(model, filepath):
    print(f"📦 Exporting weights to {filepath}...")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'wb') as f:
        # --- HEADER ---
        # Magic "AGNS", Version (1), ArchType (0=MLP), NumLayers (2)
        f.write(b'AGNS')
        f.write(struct.pack('<III', 1, 0, 2)) 
        
        # We have 2 linear layers in this model to export
        layers = [
            (model.fc1, 1), # 1 = ReLU activation follows this layer
            (model.fc2, 0)  # 0 = No activation (raw logits output)
        ]
        
        for layer, act_type in layers:
            weight = layer.weight.detach().numpy() # Shape: (out_features, in_features)
            bias = layer.bias.detach().numpy()     # Shape: (out_features)
            
            out_features, in_features = weight.shape
            
            # --- LAYER METADATA ---
            # LayerType (0=Dense), Activation, InFeatures, OutFeatures
            f.write(struct.pack('<IIII', 0, act_type, in_features, out_features))
            
            # --- LAYER WEIGHTS & BIASES ---
            # Flatten weights to 1D and pack as float32
            weight_flat = weight.flatten()
            f.write(struct.pack(f'<{len(weight_flat)}f', *weight_flat))
            f.write(struct.pack(f'<{len(bias)}f', *bias))
            
    print(f"✅ Export successful. File size: {os.path.getsize(filepath)} bytes.")

if __name__ == "__main__":
    trained_model = train_model()
    export_path = "../../data/models/v1_base_mlp.agnos"
    export_to_agnos(trained_model, export_path)