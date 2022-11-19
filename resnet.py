import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import transforms, models
from custom_image_dataset import CustomImageDataset

from dataset import Dataset
from torch.autograd import Variable

def load_and_train_model(train_dataset, target_variable_name = 'label', epochs = 10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(pretrained=True)
    # print(model)

    for param in model.parameters():
        param.requires_grad = False
        
    model.fc = nn.Sequential(nn.Linear(2048, 512),
                                    nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(512, 10),
                                    nn.LogSoftmax(dim=1))
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.003)

    model.to(device)

    train_transforms = transforms.Compose([transforms.ToPILImage(),
                                            transforms.Resize(224),
                                        transforms.ToTensor(),
                                        ])

    # train_dataset = Dataset(type='csv', file_name='result/train_processed.csv', target_variable_name='label')

    X_train = train_dataset.dataset.drop(target_variable_name, axis=1)
    y_train = train_dataset.dataset[target_variable_name]

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

    # quick hack, we copy the dataset and replace the dataset with the train and test data
    import pandas as pd
    train_dataset = Dataset(type='csv', file_name='result/train_processed.csv', target_variable_name=target_variable_name)
    # concat x_train and y_train
    train_dataset.dataset = pd.concat([X_train, y_train], axis=1)

    test_dataset = Dataset(type='csv', file_name='result/train_processed.csv', target_variable_name='label')
    # concat x_test and y_test
    test_dataset.dataset = pd.concat([X_test, y_test], axis=1)

    # custom image dtaaset for traininig with pytorch
    img_dataset_train = CustomImageDataset(train_dataset, img_dir='train_test_data/train', transform=train_transforms) # Original size
    img_dataset_test = CustomImageDataset(test_dataset, img_dir='train_test_data/test', transform=train_transforms) # Original size

        
    trainloader = torch.utils.data.DataLoader(img_dataset_train, batch_size=64)
    testloader = torch.utils.data.DataLoader(img_dataset_test, batch_size=64, num_workers=2)

    # epochs = 10
    steps = 0
    running_loss = 0
    print_every = 10
    train_losses, test_losses = [], []

    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                f1 = 0
                precission = 0
                recall = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in testloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        test_loss += batch_loss.item()
                        
                        ps = torch.exp(logps)

                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                        # avoid division by zero calculating precision
                        if torch.sum(top_class == 1) > 0:
                            precission += torch.sum((top_class == 1) & (labels.view(*top_class.shape) == 1)) / torch.sum(top_class == 1)
                        else:
                            precission += 0

                        # avoid division by zero calculating recall
                        if torch.sum(labels.view(*top_class.shape) == 1) > 0:
                            recall += torch.sum((top_class == 1) & (labels.view(*top_class.shape) == 1)) / torch.sum(labels.view(*top_class.shape) == 1)
                        else:
                            recall += 0

                        # avoid division by zero calculating f1
                        if precission + recall > 0:
                            f1 = 2 * (precission * recall ) / (precission + recall)
                        else:
                            f1 = 0

                train_losses.append(running_loss/len(trainloader))
                test_losses.append(test_loss/len(testloader))                    
                print(f"Epoch {epoch+1}/{epochs}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Test loss: {test_loss/len(testloader):.3f}.. "
                    f"Test accuracy: {accuracy/len(testloader):.3f}.. "
                    f"Test f1: {f1/len(testloader):.3f}.. "
                    f"Test precission: {precission/len(testloader):.3f}.. "
                    f"Test recall: {recall/len(testloader):.3f}")

                running_loss = 0
                model.train()

    torch.save(model, 'model.pth')


    plt.plot(train_losses, label='Training loss')
    plt.plot(test_losses, label='Validation loss')
    plt.legend(frameon=False)
    plt.show()

    # test the model
    model.eval()

    return model


def test_model(img_dataset_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model=torch.load('model.pth')
    model.eval()

    # custom image dtaaset for traininig with pytorch
    test_transforms = transforms.Compose([transforms.ToPILImage(),
                                                transforms.Resize(224),
                                            transforms.ToTensor(),
                                            ])
    
    labels_pred = []

    for image in img_dataset_test.dataset:
        image_tensor = test_transforms(image).float()
        image_tensor = image_tensor.unsqueeze_(0)
        
        input = Variable(image_tensor)
        input = input.to(device)
        
        output = model(input)
        index = output.data.cpu().numpy().argmax()

        labels_pred.append(index)

    return labels_pred