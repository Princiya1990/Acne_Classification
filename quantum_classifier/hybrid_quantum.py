import os
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.opflow import PauliSumOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

# 1. Load data
df = pd.read_csv('/data/quantum_features_with_labels.csv')
feature_cols = [col for col in df.columns if col.startswith('feat_')]
X = df[feature_cols].values
y = df['iga_class'].values.reshape(-1, 1)
X = MinMaxScaler().fit_transform(X)
enc = OneHotEncoder(sparse=False)
y_oh = enc.fit_transform(y)

# 2. Custom Dataset for PyTorch
class QuantumDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

X_train, X_test, y_train, y_test = train_test_split(X, y_oh, test_size=0.2, random_state=42)
train_set = QuantumDataset(X_train, y_train)
test_set = QuantumDataset(X_test, y_test)
train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
test_loader = DataLoader(test_set, batch_size=8, shuffle=False)

# 3. Build 10-qubit VQC circuit, use <Z> on each qubit for expectation
def create_quantum_circuit():
    n_qubits = 10
    qc = QuantumCircuit(n_qubits)
    amp_features = ParameterVector('amp', 7)
    ang_features = ParameterVector('ang', 3)
    weights = ParameterVector('theta', n_qubits)
    qc.initialize(amp_features, range(7))
    for i in range(3):
        qc.ry(ang_features[i], 7+i)
    for i in range(n_qubits-1):
        qc.cx(i, i+1)
    for i in range(n_qubits):
        qc.ry(weights[i], i)
    return qc, amp_features, ang_features, weights

qc, amp_features, ang_features, weights = create_quantum_circuit()
input_params = list(amp_features) + list(ang_features)
weight_params = list(weights)

# Observables: <Z> on each qubit
observables = [PauliSumOp.from_list([(f'Z{i}', 1)]) for i in range(10)]

qnn = EstimatorQNN(
    circuit=qc,
    observables=observables,
    input_params=input_params,
    weight_params=weight_params,
    exp_val=True,
    gradient=None
)

# 4. Build Hybrid PyTorch Model: QNN + FC layer
class HybridQNNModel(nn.Module):
    def __init__(self, qnn, n_outputs=5):
        super().__init__()
        self.qnn = TorchConnector(qnn)
        self.fc = nn.Linear(10, n_outputs)
    def forward(self, x):
        q_out = self.qnn(x)          # (batch, 10)
        out = self.fc(q_out)         # (batch, n_outputs)
        return torch.softmax(out, dim=1)

model = HybridQNNModel(qnn, n_outputs=y_oh.shape[1])

# 5. Train with PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

def train_epoch(model, loader):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        # Use class indices for CrossEntropyLoss
        target = torch.argmax(y, dim=1)
        loss = loss_fn(outputs, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def eval_accuracy(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)
            labels = torch.argmax(y, dim=1)
            correct += (preds == labels).sum().item()
            total += len(y)
    return correct / total

# Training loop
epochs = 10  # Adjust as required
for epoch in range(epochs):
    train_loss = train_epoch(model, train_loader)
    acc = eval_accuracy(model, test_loader)
    print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} - Test Accuracy: {acc:.4f}")

