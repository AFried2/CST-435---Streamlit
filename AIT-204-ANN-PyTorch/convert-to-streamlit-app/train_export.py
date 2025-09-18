# 5) Torch training (IMPROVED)
import torch
from torch.utils.data import TensorDataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NewsMLP(Xtr.shape[1], num_classes).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
crit = nn.CrossEntropyLoss()

# --- Create DataLoader to handle batching ---
batch_size = 64
train_ds = TensorDataset(torch.tensor(Xtr, dtype=torch.float32), 
                         torch.tensor(ytr, dtype=torch.long))
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

# --- Proper training loop with epochs and batches ---
print("Starting improved training...")
NUM_EPOCHS = 10 # Increased epochs
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for i, (xb, yb) in enumerate(train_loader):
        xb, yb = xb.to(device), yb.to(device)
        
        logits = model(xb)
        loss = crit(logits, yb)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        running_loss += loss.item()
    
    # Print loss for the epoch
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {running_loss / len(train_loader):.4f}")
print("Training finished.")