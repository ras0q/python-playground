import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets.mnist import MNIST

image_path = "./"
transform = transforms.Compose([transforms.ToTensor()])
mnist_train_dataset = torchvision.datasets.MNIST(
    root=image_path, train=True, transform=transform, download=True
)
mnist_test_dataset = MNIST(
    root=image_path, train=False, transform=transform, download=False
)
batch_size = 64

torch.manual_seed(1)

train_dl: DataLoader[MNIST] = DataLoader(mnist_train_dataset, batch_size, shuffle=True)

hidden_units = [32, 16]
image_size = mnist_train_dataset[0][0].shape
input_size = image_size[0] * image_size[1] * image_size[2]

# モデルの作成
all_layers: list[nn.Module] = [nn.Flatten()]
for hidden_init in hidden_units:
    all_layers.append(nn.Linear(input_size, hidden_init))
    all_layers.append(nn.ReLU())
    input_size = hidden_init
all_layers.append(nn.Linear(hidden_units[-1], 10))
all_layers.append(nn.Softmax(dim=1))
model = nn.Sequential(*all_layers)

# 損失関数の定義
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
torch.manual_seed(1)

# 訓練
num_epochs = 20
for epoch in range(num_epochs):
    accuracy_hist_train = 0.0
    for x_batch, y_batch in train_dl:
        pred = model(x_batch)
        loss = loss_fn(pred, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
        accuracy_hist_train += is_correct.sum()

    accuracy_hist_train /= len(mnist_train_dataset)
    print(f"Epoch: {epoch}, Accuracy: {accuracy_hist_train:.4f}")

pred = model(mnist_test_dataset.data / 255.0)
is_correct = (torch.argmax(pred, dim=1) == mnist_test_dataset.targets).float()
print(f"Test accuracy: {is_correct.mean():.4f}")
