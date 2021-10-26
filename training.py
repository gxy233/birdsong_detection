import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.notebook import tqdm
from model import *
from load_data import*
model = Classifier()
print(model)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
torch.backends.cudnn.benchmark = True
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)
model = to_device(Classifier(), device)
lr = 1e-5
epochs = 8
history = [evaluate(model, val_dl)]
print(history)
history += fit(epochs, lr, model, train_dl, val_dl)
history += fit(3, 1e-6, model, train_dl, val_dl)

plt.plot([x['avg_loss'] for x in history])
plt.title('Losses over epochs')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()


plt.plot([x['avg_acc'] for x in history])
plt.title('Accuracy over epochs')
plt.xlabel('epochs')
plt.ylabel('acc')
plt.show()


torch.save(model.state_dict(), 'Classifier.pth')
model.load_state_dict(torch.load('Classifier.pth'))
evaluate(model, val_dl)

evaluate(model, val_dl)