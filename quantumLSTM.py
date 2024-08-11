import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the dataset
ddos_data = pd.read_csv('Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')

# Basic preprocessing
# Assuming the dataset has a target column named 'Label' which we need to predict
# and that all other columns are features. Modify according to the actual dataset.
features = ddos_data.drop(columns=['Label'])
target = ddos_data['Label'].apply(lambda x: 1 if x == 'ddos' else 0)  # Convert to binary target

# Normalize the features
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Convert to torch tensors
features = torch.tensor(features, dtype=torch.float32)
target = torch.tensor(target.values, dtype=torch.float32).unsqueeze(1)

# Split the data into sequences
sequence_length = 10  # Length of each time series sequence
def create_sequences(features, target, seq_length):
    sequences = []
    labels = []
    for i in range(len(features) - seq_length):
        seq = features[i:i+seq_length]
        label = target[i+seq_length]
        sequences.append(seq)
        labels.append(label)
    return torch.stack(sequences), torch.stack(labels)

# Create sequences from the dataset
sequences, labels = create_sequences(features, target, sequence_length)

# Split into train and test sets
train_features, test_features, train_labels, test_labels = train_test_split(sequences, labels, test_size=0.2, random_state=42)

# Convert to DataLoader for batching
train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
test_dataset = torch.utils.data.TensorDataset(test_features, test_labels)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)



class AdvancedQuantumLSTMCell(nn.Module):
    def __init__(self, n_qubits, n_layers):
        super(AdvancedQuantumLSTMCell, self).__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Initialize quantum circuit parameters
        self.theta = nn.Parameter(0.01 * torch.randn(n_layers, n_qubits, 3))

        # Define quantum device
        self.dev = qml.device("default.qubit", wires=self.n_qubits)

        def qnode(inputs, hidden):
            qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
            for i in range(n_layers):
                qml.templates.StronglyEntanglingLayers(hidden[i], wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.qnode = qml.QNode(qnode, self.dev, interface="torch")

    def forward(self, inputs, hidden):
        quantum_out = self.qnode(inputs, hidden)
        hidden_out = torch.tanh(quantum_out)
        return hidden_out, hidden_out

# Hybrid LSTM Network with Classical and Quantum Layers
class AdvancedLSTMNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_qubits, n_layers):
        super(AdvancedLSTMNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Quantum LSTM cell
        self.quantum_lstm = AdvancedQuantumLSTMCell(n_qubits, n_layers)

        # Classical LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initial hidden and cell states for LSTM
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        c0 = torch.zeros(1, x.size(0), self.hidden_size)

        # Classical LSTM output
        out, _ = self.lstm(x, (h0, c0))

        # Quantum LSTM processing on the last time step
        quantum_out, _ = self.quantum_lstm(out[:, -1, :], out[:, -1, :].unsqueeze(0))

        # Fully connected layer to produce final output
        out = self.fc(quantum_out)
        return out

# Custom Hybrid Loss Function
class HybridLoss(nn.Module):
    def __init__(self, classical_loss_weight=0.5):
        super(HybridLoss, self).__init__()
        self.classical_loss_weight = classical_loss_weight
        self.mse_loss = nn.MSELoss()

    def forward(self, classical_output, quantum_output, target):
        classical_loss = self.mse_loss(classical_output, target)
        quantum_loss = self.mse_loss(quantum_output, target)
        total_loss = (self.classical_loss_weight * classical_loss +
                      (1 - self.classical_loss_weight) * quantum_loss)
        return total_loss



# Example usage
n_qubits = 8
n_layers = 2
input_size = train_features.shape[-1]  # Number of features
hidden_size = 8  # Should match the number of qubits
output_size = 1  # Predicting a single value (DDoS or not)

model = AdvancedLSTMNetwork(input_size, hidden_size, output_size, n_qubits, n_layers)
criterion = HybridLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 20
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_idx, (batch_features, batch_labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(batch_features)
        loss = criterion(output, output, batch_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader)}")

# Test the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch_features, batch_labels in test_loader:
        output = model(batch_features)
        predicted = (output > 0.5).float()  # Threshold for binary classification
        total += batch_labels.size(0)
        correct += (predicted == batch_labels).sum().item()

print(f'Accuracy: {100 * correct / total}%')

