import torch
from torch.utils.data import DataLoader, TensorDataset


def train_model(
    model,
    X_train,
    y_train,
    epochs=10,
    batch_size=256,
    learning_rate=1e-3,
    device="cpu",
):
    model = model.to(device)

    X_tensor = torch.tensor(
        X_train, dtype=torch.float32
    )
    y_tensor = torch.tensor(
        y_train, dtype=torch.long
    )

    dataset = TensorDataset(X_tensor, y_tensor)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
    )

    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / max(len(loader), 1)

        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"Loss: {average_loss:.4f}"
        )

    return model
