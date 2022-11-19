import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from custom_image_dataset import CustomImageDataset

from dataset import Dataset
from torch.autograd import Variable

classes = ('0', '1', '2')

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(6, 166, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)
	
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        print("x", x.shape)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = Network()
# model.init()
### Optimizaion perdida ####

from torch.optim import Adam

loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)


######  guardar modelo ####
def saveModel():
    path = "./myFirstModel.pth"
    torch.save(model.state_dict(), path)

####### test f1 score ####
def testF1Score(test_loader):
    
    model.eval()
    
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # iniciamos modelo
            outputs = model(images)
            # obtenemos los valores predecidos
            _, predicted = torch.max(outputs.data, 1)
    
    f1_score = f1_score(labels, predicted) 
    return(f1_score)

############# Funcion de train ######### 
def train(num_epochs):
    
    best_f1_score = 0.0

    train_dataset = Dataset(type='csv', file_name='result/train_processed.csv', target_variable_name='label')
    
    # annotations_file, img_dir, transform=None, target_transform=None
    from torchvision.transforms import transforms
    transformations = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    img_dataset_train = CustomImageDataset(train_dataset, img_dir='train_test_data/train', transform=transformations) # Original size
    img_dataset_test = CustomImageDataset(train_dataset, img_dir='train_test_data/test', transform=transformations) # Original size

    print(len(train_dataset.dataset))

    # mira si hay GPU de nvidia
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")
    model.to(device)
    
    train_loader = torch.utils.data.DataLoader(img_dataset_train, batch_size=4)
    test_loader = torch.utils.data.DataLoader(img_dataset_test, batch_size=4, num_workers=2)

    for epoch in range(num_epochs):  # loop de las epoca 
        running_loss = 0.0
        running_acc = 0.0

        for i, (images, labels) in enumerate(train_loader, 0):
            # get the inputs
            images = Variable(images.to(device))
            labels = Variable(labels.to(device))

            # zero the parameter gradients
            optimizer.zero_grad()
            # predict classes using images from the training set
            outputs = model(images)
            
            # compute the loss based on model output and real labels
            print(len(outputs))
            print(len(labels))

            loss = loss_fn(outputs, labels)
            # backpropagate the loss
            loss.backward()
            # adjust parameters based on the calculated gradients
            optimizer.step()

            # Let's print statistics for every 1,000 images
            running_loss += loss.item()     # extract the loss value
            if i % 1000 == 999:    
                # print every 1000 (twice per epoch) 
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 1000))
                # zero the loss
                running_loss = 0.0

        # Compute and print the average accuracy fo this epoch when tested over all 10000 test images
        F1Score = testF1Score(test_loader)
        accuracy = 100 * F1Score
        print('For epoch', epoch+1,'the test accuracy over the whole test set is %d %%' % (accuracy))
        
        # we want to save the model if the accuracy is the best
        if F1Score > best_f1_score:
            saveModel()
            best_f1_score = F1Score

# def testBatch():
#     # get batch of images from the test DataLoader  
#     images, labels = next(iter(test_loader))

#     # show all images as one image grid
#     imageshow(torchvision.utils.make_grid(images))
   
#     # Show the real labels on the screen 
#     print('Real labels: ', ' '.join('%5s' % classes[labels[j]] 
#                                for j in range(batch_size)))
  
#     # Let's see what if the model identifiers the  labels of those example
#     outputs = model(images)
    
#     # We got the probability for every 10 labels. The highest (max) probability should be correct label
#     _, predicted = torch.max(outputs, 1)
    
#     # Let's show the predicted labels on the screen to compare with the real ones
#     print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] 
#                               for j in range(batch_size)))

if __name__ == '__main__':
    import torch
    torch.cuda.empty_cache()

    train(5)

    # model = Network()
    # model.load_state_dict(torch.load('./myFirstModel.pth'))
    # model.eval()