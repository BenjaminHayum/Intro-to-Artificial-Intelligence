import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.


def get_data_loader(training=True):
    """
    TODO: implement this function.

    INPUT:
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """
    custom_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    if training == True:
        train_set = datasets.FashionMNIST("./data", train=True, download=True, transform=custom_transform)
        loader = torch.utils.data.DataLoader(train_set, batch_size=64)
    else:
        test_set = datasets.FashionMNIST("./data", train=False, transform=custom_transform)
        loader = torch.utils.data.DataLoader(test_set, batch_size=64)
    return loader


def build_model():
    """
    TODO: implement this function.

    INPUT:
        None

    RETURNS:
        An untrained neural network model
    """
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
        # , nn.Softmax()
    )
    return model


def train_model(model, train_loader, criterion, T):
    """
    TODO: implement this function.

    INPUT:
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy
        T - number of epochs for training

    RETURNS:
        None
    """
    opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    model.train()
    num_epochs = T
    for epoch in range(num_epochs):
        epoch_total_accurate = 0
        epoch_total_loss = 0
        num_batches = 0
        num_data_points = 0
        for inputs, targets in train_loader:
            num_batches += 1
            num_data_points += targets.size(dim=0)
            # Calculate Outputs
            predictions = model(inputs)
            numpy_predictions = predictions.detach().numpy()
            predicted_classes = np.argmax(numpy_predictions, axis=1)
            # Use argmax for each of the 64 data poin s

            # Calculate Accuracy
            numpy_targets = targets.detach().numpy()
            epoch_total_accurate += (predicted_classes == numpy_targets).sum()

            # Calculate Loss
            loss = criterion(predictions, targets)
            epoch_total_loss += loss.item()

            # Backpropagate
            opt.zero_grad()
            loss.backward()
            opt.step()

        epoch_string = "Train Epoch: " + str(epoch)
        accuracy_percentage = (epoch_total_accurate / num_data_points) * 100
        accuracy_string = "Accuracy: " + str(epoch_total_accurate) + "/" + str(num_data_points) + "(" + str(
            round(accuracy_percentage, 2)) + "%)"
        loss_string = "Loss: " + str(round(epoch_total_loss / num_batches, 4))
        print(epoch_string + "\t" + accuracy_string + "\t" + loss_string)


def evaluate_model(model, test_loader, criterion, show_loss=True):
    """
    TODO: implement this function.

    INPUT:
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy

    RETURNS:
        None
    """
    model.eval()
    with torch.no_grad():
        total_accurate = 0
        total_loss = 0
        num_batches = 0
        num_data_points = 0
        for inputs, targets in test_loader:
            num_batches += 1
            num_data_points += targets.size(dim=0)
            # Calculate Outputs
            predictions = model(inputs)
            numpy_predictions = predictions.detach().numpy()
            predicted_classes = np.argmax(numpy_predictions, axis=1)
            # Use argmax for each of the 64 data points

            # Calculate Accuracy
            numpy_targets = targets.detach().numpy()
            total_accurate += (predicted_classes == numpy_targets).sum()

            # Calculate Loss
            loss = criterion(predictions, targets)
            total_loss += loss.item()

    accuracy_percentage = (total_accurate / num_data_points) * 100
    accuracy_string = "Accuracy: " + str(round(accuracy_percentage, 2)) + "%"
    loss_string = "Average Loss: " + str(round(total_loss / num_batches, 4))
    if show_loss == True:
        print(accuracy_string + "\n" + loss_string)
    else:
        print(accuracy_string)


def predict_label(model, test_images, index):
    """
    TODO: implement this function.

    INPUT:
        model - the trained model
        test_images   -  test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
    """
    class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag",
                   "Ankle Boot"]
    image = test_images[index]
    model.eval()
    prediction = model(image)
    probabilities = F.softmax(prediction, dim=1)
    first_max = -3
    first_index = -1
    second_max = -2
    second_index = -1
    third_max = -1
    third_index = -1
    probabilities = probabilities[0]
    for i in range(len(probabilities)):
        prob = float(probabilities[i].detach().numpy())
        if prob > first_max:
            first_max = prob
            first_index = i
    for i in range(len(probabilities)):
        prob = float(probabilities[i].detach().numpy())
        if prob > second_max and i != first_index:
            second_max = prob
            second_index = i
    for i in range(len(probabilities)):
        prob = float(probabilities[i].detach().numpy())
        if prob > third_max and i != second_index and i != first_index:
            third_max = prob
            third_index = i
    first_class = class_names[first_index]
    second_class = class_names[second_index]
    third_class = class_names[third_index]
    print(str(first_class) + ": " + str(round(100 * first_max, 2)) + "%")
    print(str(second_class) + ": " + str(round(100 * second_max, 2)) + "%")
    print(str(third_class) + ": " + str(round(100 * third_max, 2)) + "%")


if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    '''
    # Data Loader Tests
    train_loader = get_data_loader()
    test_loader = get_data_loader(False)

    # Build Model Tests
    model = build_model()

    # Train Model Tests
    criterion = nn.CrossEntropyLoss()
    num_epochs = 5
    train_model(model, train_loader, criterion, num_epochs)

    # Evaluate Model Tests
    evaluate_model(model, test_loader, criterion, show_loss=False)
    evaluate_model(model, test_loader, criterion, show_loss=True)

    # Predict the Label Tests
    prediction_set, _ = next(iter(test_loader))
    index = 15
    predict_label(model, prediction_set, index)
