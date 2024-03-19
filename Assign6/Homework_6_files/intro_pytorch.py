import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms



def get_data_loader(training = True):
    cust_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    data_set = datasets.FashionMNIST('./data', train=training,
                                     download=True, transform=cust_transform)
    loader = torch.utils.data.DataLoader(data_set, batch_size=64)
    return loader

def build_model():
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28,128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    return model


def train_model(model, train_loader, criterion, T):
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(T):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            opt.zero_grad()

            outputs = model(inputs)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()

            running_loss += loss.item()
        accuracy = 100 * correct / total
        avg_loss = running_loss / len(train_loader)
        print('Train Epoch: {} Accuracy: {}/{}({:.2f}%) Loss: {:.3f}'.format(
            epoch, correct, total, accuracy, avg_loss))


def evaluate_model(model, test_loader, criterion, show_loss = True):
    model.eval()
    running_loss = 0.0
    accurate = 0
    total = 0
    with torch.no_grad():
        for input, label in test_loader:
            output = model(input)
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            accurate += (predicted == label).sum().item()
            loss = criterion(output, label)
            running_loss += loss.item()
    if show_loss:
        print("Average loss: {:.4f}".format(running_loss/len(test_loader.dataset)))    
    print("Accuracy: {:.2f}%".format(100*accurate/total))


def predict_label(model, test_images, index):
    class_names = ['T-shirt/top','Trouser','Pullover','Dress',
                   'Coat','Sandal','Shirt','Sneaker','Bag',
                   'Ankle Boot']
    logits = model(test_images[index])
    prob = F.softmax(logits, dim=1)
    prob_list = prob.tolist()
    flat_list = [item for sublist in prob_list for item in sublist]
    map = {class_names[i]: flat_list[i]*100 for i in range(len(class_names))}
    sorted_map = sorted(map.items(), key=lambda x:x[1], reverse=True)
    for i in range(3):
        print(f'{sorted_map[i][0]}: {sorted_map[i][1]:.2f}%')

if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    '''
    criterion = nn.CrossEntropyLoss()
