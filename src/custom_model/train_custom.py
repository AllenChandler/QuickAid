import torch
import torch.optim as optim
from simple_model import SimpleCNN
from data_loader import get_data_loaders
from losses import get_loss_function


import os
import pandas as pd
import matplotlib.pyplot as plt

# Remove any existing model weights to ensure training starts fresh
if os.path.exists("models/custom/custom_model.pt"):
    os.remove("models/custom/custom_model.pt")

# Early stopping parameters
early_stopping_patience = 5
best_valid_loss = float("inf")
patience_counter = 0

# Lists to store loss values for logging
train_losses = []
valid_losses = []

def train_model(data_dir, num_classes, num_epochs=20, batch_size=32, learning_rate=0.001):
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model, data loaders, loss function, and optimizer
    model = SimpleCNN(num_classes=num_classes).to(device)
    train_loader, valid_loader = get_data_loaders(data_dir, batch_size)
    criterion = get_loss_function()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Track the best validation loss
    best_loss = float("inf")

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Calculate average train loss
        train_loss /= len(train_loader)

        # Validation loop
        model.eval()
        valid_loss = 0.0  # Initialize valid_loss before the validation loop
        if len(valid_loader) > 0:  # Ensure validation loader has data
            with torch.no_grad():
                for inputs, labels in valid_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    valid_loss += loss.item()
            valid_loss /= len(valid_loader)  # Calculate average validation loss
        else:
            valid_loss = float("inf")  # Default to a high value if no validation data exists

        # Log the losses for this epoch
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")

        # Early stopping logic
        global best_valid_loss, patience_counter
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break

        # Save the best model
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), f"models/custom/custom_model.pt")
            print("Saved Best Model!")

    # Save losses to an Excel file
    loss_data = {
        "Epoch": list(range(1, len(train_losses) + 1)),
        "Train Loss": train_losses,
        "Valid Loss": valid_losses
    }
    df = pd.DataFrame(loss_data)
    excel_path = "QuickAid_loss_log.xlsx"
    df.to_excel(excel_path, index=False)
    print(f"Loss data saved to {excel_path}")

    # Generate a line chart for loss progression
    plt.figure()
    plt.plot(df["Epoch"], df["Train Loss"], label="Train Loss", marker="o")
    plt.plot(df["Epoch"], df["Valid Loss"], label="Valid Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    chart_path = "QuickAid_loss_chart.png"
    plt.savefig(chart_path)
    print(f"Loss chart saved to {chart_path}")

if __name__ == "__main__":
    train_model(data_dir="data/images", num_classes=7)
