import streamlit as st
from PIL import Image
import numpy as np
from parsers import args
import torch
import torchvision.transforms as transforms
import resnet_cifar as model_cifar
import pickle
import torch
import torch.nn as nn

# Load your pre-trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# Function to preprocess the image
def preprocess_image(image):
    # Resize the image to the size required by your model
    transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    ])

    input_image = transform(image).unsqueeze(0) 
    
    return input_image

def CUBpreprocess_image(image):
    # Resize the image to the size required by your model
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    input_image= transform(image).unsqueeze(0)
    return input_image

# Function to make predictions
def predict(image,model):
    # Preprocess the image
    processed_image = preprocess_image(image)
    # Make predictions
    with torch.no_grad():
        model.eval()  
    
        predictions = model(processed_image)
    return predictions

def minipredict0(image,model):
    image = CUBpreprocess_image(image)
    model.eval()
    new_model = torch.nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2,
                                            model.layer3, model.layer4, model.avgpool, Flatten())

    model_linear = torch.load("ACIL_CUB_weights_linear_0_new.pth",map_location='cpu')
        #print(summary(new_model,input_size=(3,32,32)))

    new_model.eval()
    with torch.no_grad():
        X_0_cnn = new_model(image)
        linear = torch.matmul(X_0_cnn,model_linear.t())
        Rel = torch.nn.ReLU()
        relu = Rel(linear)
        weights = torch.load("ACIL_CUB_weights0_new.pth",map_location='cpu')
        #print(relu.size(),weights.size())
        output = torch.matmul(relu,weights.t())
        print(output)
    return output

def minipredict1(image,model):
    image = CUBpreprocess_image(image)
    model.eval()
    new_model = torch.nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2,
                                            model.layer3, model.layer4, model.avgpool, Flatten())

    model_linear = torch.load("ACIL_CUB_weights_linear_0_new.pth",map_location='cpu')
        #print(summary(new_model,input_size=(3,32,32)))

    new_model.eval()
    with torch.no_grad():
        X_0_cnn = new_model(image)
        linear = torch.matmul(X_0_cnn,model_linear.t())
        Rel = torch.nn.ReLU()
        relu = Rel(linear)
        weights = torch.load("ACIL_CUB_weights0_new.pth",map_location='cpu')
        #print(relu.size(),weights.size())
        output = torch.matmul(relu,weights.t())
        print(output)
    return output

def minipredict2(image,model):
    image = CUBpreprocess_image(image)
    model.eval()
    new_model = torch.nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2,
                                            model.layer3, model.layer4, model.avgpool, Flatten())

    model_linear = torch.load("ACIL_CUB_weights_linear_0_new.pth",map_location='cpu')
        #print(summary(new_model,input_size=(3,32,32)))

    new_model.eval()
    with torch.no_grad():
        X_0_cnn = new_model(image)
        linear = torch.matmul(X_0_cnn,model_linear.t())
        Rel = torch.nn.ReLU()
        relu = Rel(linear)
        weights = torch.load("ACIL_CUB_weights0_new.pth",map_location='cpu')
        #print(relu.size(),weights.size())
        output = torch.matmul(relu,weights.t())
        print(output)
    return output

def minipredict3(image,model):
    image = CUBpreprocess_image(image)
    model.eval()
    new_model = torch.nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2,
                                            model.layer3, model.layer4, model.avgpool, Flatten())

    model_linear = torch.load("ACIL_CUB_weights_linear_0_new.pth",map_location='cpu')
        #print(summary(new_model,input_size=(3,32,32)))

    new_model.eval()
    with torch.no_grad():
        X_0_cnn = new_model(image)
        linear = torch.matmul(X_0_cnn,model_linear.t())
        Rel = torch.nn.ReLU()
        relu = Rel(linear)
        weights = torch.load("ACIL_CUB_weights0_new.pth",map_location='cpu')
        #print(relu.size(),weights.size())
        output = torch.matmul(relu,weights.t())
        print(output)
    return output

def minipredict4(image,model):
    image = CUBpreprocess_image(image)
    model.eval()
    new_model = torch.nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2,
                                            model.layer3, model.layer4, model.avgpool, Flatten())

    model_linear = torch.load("ACIL_CUB_weights_linear_0_new.pth",map_location='cpu')
        #print(summary(new_model,input_size=(3,32,32)))

    new_model.eval()
    with torch.no_grad():
        X_0_cnn = new_model(image)
        linear = torch.matmul(X_0_cnn,model_linear.t())
        Rel = torch.nn.ReLU()
        relu = Rel(linear)
        weights = torch.load("ACIL_CUB_weights0_new.pth",map_location='cpu')
        #print(relu.size(),weights.size())
        output = torch.matmul(relu,weights.t())
        print(output)
    return output

def minipredict5(image,model):
    image = CUBpreprocess_image(image)
    model.eval()
    new_model = torch.nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2,
                                            model.layer3, model.layer4, model.avgpool, Flatten())

    model_linear = torch.load("ACIL_CUB_weights_linear_0_new.pth",map_location='cpu')
        #print(summary(new_model,input_size=(3,32,32)))

    new_model.eval()
    with torch.no_grad():
        X_0_cnn = new_model(image)
        linear = torch.matmul(X_0_cnn,model_linear.t())
        Rel = torch.nn.ReLU()
        relu = Rel(linear)
        weights = torch.load("ACIL_CUB_weights0_new.pth",map_location='cpu')
        #print(relu.size(),weights.size())
        output = torch.matmul(relu,weights.t())
        print(output)
    return output

def minipredict6(image,model):
    image = CUBpreprocess_image(image)
    model.eval()
    new_model = torch.nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2,
                                            model.layer3, model.layer4, model.avgpool, Flatten())

    model_linear = torch.load("ACIL_CUB_weights_linear_0_new.pth",map_location='cpu')
        #print(summary(new_model,input_size=(3,32,32)))

    new_model.eval()
    with torch.no_grad():
        X_0_cnn = new_model(image)
        linear = torch.matmul(X_0_cnn,model_linear.t())
        Rel = torch.nn.ReLU()
        relu = Rel(linear)
        weights = torch.load("ACIL_CUB_weights0_new.pth",map_location='cpu')
        #print(relu.size(),weights.size())
        output = torch.matmul(relu,weights.t())
        print(output)
    return output

def minipredict7(image,model):
    image = CUBpreprocess_image(image)
    model.eval()
    new_model = torch.nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2,
                                            model.layer3, model.layer4, model.avgpool, Flatten())

    model_linear = torch.load("ACIL_CUB_weights_linear_0_new.pth",map_location='cpu')
        #print(summary(new_model,input_size=(3,32,32)))

    new_model.eval()
    with torch.no_grad():
        X_0_cnn = new_model(image)
        linear = torch.matmul(X_0_cnn,model_linear.t())
        Rel = torch.nn.ReLU()
        relu = Rel(linear)
        weights = torch.load("ACIL_CUB_weights0_new.pth",map_location='cpu')
        #print(relu.size(),weights.size())
        output = torch.matmul(relu,weights.t())
        print(output)
    return output

def minipredict8(image,model):
    image = CUBpreprocess_image(image)
    model.eval()
    new_model = torch.nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2,
                                            model.layer3, model.layer4, model.avgpool, Flatten())

    model_linear = torch.load("ACIL_CUB_weights_linear_0_new.pth",map_location='cpu')
        #print(summary(new_model,input_size=(3,32,32)))

    new_model.eval()
    with torch.no_grad():
        X_0_cnn = new_model(image)
        linear = torch.matmul(X_0_cnn,model_linear.t())
        Rel = torch.nn.ReLU()
        relu = Rel(linear)
        weights = torch.load("ACIL_CUB_weights0_new.pth",map_location='cpu')
        #print(relu.size(),weights.size())
        output = torch.matmul(relu,weights.t())
        print(output)
    return output

def minipredict9(image,model):
    image = CUBpreprocess_image(image)
    model.eval()
    new_model = torch.nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2,
                                            model.layer3, model.layer4, model.avgpool, Flatten())

    model_linear = torch.load("ACIL_CUB_weights_linear_0_new.pth",map_location='cpu')
        #print(summary(new_model,input_size=(3,32,32)))

    new_model.eval()
    with torch.no_grad():
        X_0_cnn = new_model(image)
        linear = torch.matmul(X_0_cnn,model_linear.t())
        Rel = torch.nn.ReLU()
        relu = Rel(linear)
        weights = torch.load("ACIL_CUB_weights0_new.pth",map_location='cpu')
        #print(relu.size(),weights.size())
        output = torch.matmul(relu,weights.t())
        print(output)
    return output
def CUBpredict0(image,model):
    image = CUBpreprocess_image(image)
    model.eval()
    new_model = torch.nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2,
                                            model.layer3, model.layer4, model.avgpool, Flatten())

    model_linear = torch.load("ACIL_CUB_weights_linear_0_new.pth",map_location='cpu')
        #print(summary(new_model,input_size=(3,32,32)))

    new_model.eval()
    with torch.no_grad():
        X_0_cnn = new_model(image)
        linear = torch.matmul(X_0_cnn,model_linear.t())
        Rel = torch.nn.ReLU()
        relu = Rel(linear)
        weights = torch.load("ACIL_CUB_weights0_new.pth",map_location='cpu')
        #print(relu.size(),weights.size())
        output = torch.matmul(relu,weights.t())
        print(output)
    return output

def CUBpredict1(image,model):
    image = CUBpreprocess_image(image)
    model.eval()
    new_model = torch.nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2,
                                            model.layer3, model.layer4, model.avgpool, Flatten())

    model_linear = torch.load("ACIL_CUB_weights_linear_1_new.pth",map_location='cpu')
        #print(summary(new_model,input_size=(3,32,32)))

    new_model.eval()
    with torch.no_grad():
        X_0_cnn = new_model(image)
        linear = torch.matmul(X_0_cnn,model_linear.t())
        Rel = torch.nn.ReLU()
        relu = Rel(linear)
        weights = torch.load("ACIL_CUB_weights1_new.pth",map_location='cpu')
        #print(relu.size(),weights.size())
        output = torch.matmul(relu,weights.t())
        print(output)
    return output

def CUBpredict2(image,model):
    image = CUBpreprocess_image(image)
    model.eval()
    new_model = torch.nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2,
                                            model.layer3, model.layer4, model.avgpool, Flatten())

    model_linear = torch.load("ACIL_CUB_weights_linear_2_new.pth",map_location='cpu')
        #print(summary(new_model,input_size=(3,32,32)))

    new_model.eval()
    with torch.no_grad():
        X_0_cnn = new_model(image)
        linear = torch.matmul(X_0_cnn,model_linear.t())
        Rel = torch.nn.ReLU()
        relu = Rel(linear)
        weights = torch.load("ACIL_CUB_weights2_new.pth",map_location='cpu')
        #print(relu.size(),weights.size())
        output = torch.matmul(relu,weights.t())
        print(output)
    return output

def CUBpredict3(image,model):
    image = CUBpreprocess_image(image)
    model.eval()
    new_model = torch.nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2,
                                            model.layer3, model.layer4, model.avgpool, Flatten())

    model_linear = torch.load("ACIL_CUB_weights_linear_3_new.pth",map_location='cpu')
        #print(summary(new_model,input_size=(3,32,32)))

    new_model.eval()
    with torch.no_grad():
        X_0_cnn = new_model(image)
        linear = torch.matmul(X_0_cnn,model_linear.t())
        Rel = torch.nn.ReLU()
        relu = Rel(linear)
        weights = torch.load("ACIL_CUB_weights3_new.pth",map_location='cpu')
        #print(relu.size(),weights.size())
        output = torch.matmul(relu,weights.t())
        print(output)
    return output

def CUBpredict4(image,model):
    image = CUBpreprocess_image(image)
    model.eval()
    new_model = torch.nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2,
                                            model.layer3, model.layer4, model.avgpool, Flatten())

    model_linear = torch.load("ACIL_CUB_weights_linear_4_new.pth",map_location='cpu')
        #print(summary(new_model,input_size=(3,32,32)))

    new_model.eval()
    with torch.no_grad():
        X_0_cnn = new_model(image)
        linear = torch.matmul(X_0_cnn,model_linear.t())
        Rel = torch.nn.ReLU()
        relu = Rel(linear)
        weights = torch.load("ACIL_CUB_weights4_new.pth",map_location='cpu')
        #print(relu.size(),weights.size())
        output = torch.matmul(relu,weights.t())
        print(output)
    return output


def CUBpredict5(image,model):
    image = CUBpreprocess_image(image)
    model.eval()
    new_model = torch.nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2,
                                            model.layer3, model.layer4, model.avgpool, Flatten())

    model_linear = torch.load("ACIL_CUB_weights_linear_5_new.pth",map_location='cpu')
        #print(summary(new_model,input_size=(3,32,32)))

    new_model.eval()
    with torch.no_grad():
        X_0_cnn = new_model(image)
        linear = torch.matmul(X_0_cnn,model_linear.t())
        Rel = torch.nn.ReLU()
        relu = Rel(linear)
        weights = torch.load("ACIL_CUB_weights5_new.pth",map_location='cpu')
        #print(relu.size(),weights.size())
        output = torch.matmul(relu,weights.t())
        print(output)
    return output

def CUBpredict6(image,model):
    image = CUBpreprocess_image(image)
    model.eval()
    new_model = torch.nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2,
                                            model.layer3, model.layer4, model.avgpool, Flatten())

    model_linear = torch.load("ACIL_CUB_weights_linear_6_new.pth",map_location='cpu')
        #print(summary(new_model,input_size=(3,32,32)))

    new_model.eval()
    with torch.no_grad():
        X_0_cnn = new_model(image)
        linear = torch.matmul(X_0_cnn,model_linear.t())
        Rel = torch.nn.ReLU()
        relu = Rel(linear)
        weights = torch.load("ACIL_CUB_weights6_new.pth",map_location='cpu')
        #print(relu.size(),weights.size())
        output = torch.matmul(relu,weights.t())
        print(output)
    return output

def CUBpredict7(image,model):
    image = CUBpreprocess_image(image)
    model.eval()
    new_model = torch.nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2,
                                            model.layer3, model.layer4, model.avgpool, Flatten())

    model_linear = torch.load("ACIL_CUB_weights_linear_7_new.pth",map_location='cpu')
        #print(summary(new_model,input_size=(3,32,32)))

    new_model.eval()
    with torch.no_grad():
        X_0_cnn = new_model(image)
        linear = torch.matmul(X_0_cnn,model_linear.t())
        Rel = torch.nn.ReLU()
        relu = Rel(linear)
        weights = torch.load("ACIL_CUB_weights7_new.pth",map_location='cpu')
        #print(relu.size(),weights.size())
        output = torch.matmul(relu,weights.t())
        print(output)
    return output

def CUBpredict8(image,model):
    image = CUBpreprocess_image(image)
    model.eval()
    new_model = torch.nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2,
                                            model.layer3, model.layer4, model.avgpool, Flatten())

    model_linear = torch.load("ACIL_CUB_weights_linear_8_new.pth",map_location='cpu')
        #print(summary(new_model,input_size=(3,32,32)))

    new_model.eval()
    with torch.no_grad():
        X_0_cnn = new_model(image)
        linear = torch.matmul(X_0_cnn,model_linear.t())
        Rel = torch.nn.ReLU()
        relu = Rel(linear)
        weights = torch.load("ACIL_CUB_weights8_new.pth",map_location='cpu')
        #print(relu.size(),weights.size())
        output = torch.matmul(relu,weights.t())
        print(output)
    return output

def CUBpredict9(image,model):
    image = CUBpreprocess_image(image)
    model.eval()
    new_model = torch.nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2,
                                            model.layer3, model.layer4, model.avgpool, Flatten())

    model_linear = torch.load("ACIL_CUB_weights_linear_9_new.pth",map_location='cpu')
        #print(summary(new_model,input_size=(3,32,32)))

    new_model.eval()
    with torch.no_grad():
        X_0_cnn = new_model(image)
        linear = torch.matmul(X_0_cnn,model_linear.t())
        Rel = torch.nn.ReLU()
        relu = Rel(linear)
        weights = torch.load("ACIL_CUB_weights9_new.pth",map_location='cpu')
        #print(relu.size(),weights.size())
        output = torch.matmul(relu,weights.t())
        print(output)
    return output



def predict0(image,model):
    model.layer4 = nn.Sequential()
    model.maxpool = nn.Sequential()
    image = preprocess_image(image)


    new_model = torch.nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2,
                                        model.layer3, model.layer4, model.avgpool, Flatten())

    model_linear = torch.load("ACIL_weights_linear_0_new.pth",map_location='cpu')
    #print(summary(new_model,input_size=(3,32,32)))

    new_model.eval() 
    with torch.no_grad():
           

        X_0_cnn = new_model(image)
        linear = torch.matmul(X_0_cnn,model_linear.t())
        Rel = torch.nn.ReLU() 
        relu = Rel(linear)
        weights = torch.load("ACIL_weights0_new.pth",map_location='cpu')
        print(relu.size(),weights.size())
        output = torch.matmul(relu,weights.t())
    return output
def predict1(image,model):
    model.layer4 = nn.Sequential()
    model.maxpool = nn.Sequential()
    image = preprocess_image(image)


    new_model = torch.nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2,
                                        model.layer3, model.layer4, model.avgpool, Flatten())

    model_linear = torch.load("ACIL_weights_linear_1_new.pth",map_location='cpu')
    #print(summary(new_model,input_size=(3,32,32)))

    new_model.eval() 
    with torch.no_grad():
           

        X_0_cnn = new_model(image)
        linear = torch.matmul(X_0_cnn,model_linear.t())
        Rel = torch.nn.ReLU() 
        relu = Rel(linear)
        weights = torch.load("ACIL_weights1_new.pth",map_location='cpu')
        print(relu.size(),weights.size())
        output = torch.matmul(relu,weights.t())
    return output
def predict2(image,model):
    model.layer4 = nn.Sequential()
    model.maxpool = nn.Sequential()
    image = preprocess_image(image)


    new_model = torch.nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2,
                                        model.layer3, model.layer4, model.avgpool, Flatten())

    model_linear = torch.load("ACIL_weights_linear_2_new.pth",map_location='cpu')
    #print(summary(new_model,input_size=(3,32,32)))

    new_model.eval() 
    with torch.no_grad():
           

        X_0_cnn = new_model(image)
        linear = torch.matmul(X_0_cnn,model_linear.t())
        Rel = torch.nn.ReLU() 
        relu = Rel(linear)
        weights = torch.load("ACIL_weights2_new.pth",map_location='cpu')
        print(relu.size(),weights.size())
        output = torch.matmul(relu,weights.t())


    return output

def predict3(image,model):
    model.layer4 = nn.Sequential()
    model.maxpool = nn.Sequential()
    image = preprocess_image(image)


    new_model = torch.nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2,
                                        model.layer3, model.layer4, model.avgpool, Flatten())

    model_linear = torch.load("ACIL_weights_linear_3_new.pth",map_location='cpu')
    #print(summary(new_model,input_size=(3,32,32)))

    new_model.eval() 
    with torch.no_grad():
           

        X_0_cnn = new_model(image)
        linear = torch.matmul(X_0_cnn,model_linear.t())
        Rel = torch.nn.ReLU() 
        relu = Rel(linear)
        weights = torch.load("ACIL_weights3_new.pth",map_location='cpu')
        print(relu.size(),weights.size())
        output = torch.matmul(relu,weights.t())
    return output

def predict4(image,model):
    model.layer4 = nn.Sequential()
    model.maxpool = nn.Sequential()
    image = preprocess_image(image)


    new_model = torch.nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2,
                                        model.layer3, model.layer4, model.avgpool, Flatten())

    model_linear = torch.load("ACIL_weights_linear_4_new.pth",map_location='cpu')
    #print(summary(new_model,input_size=(3,32,32)))

    new_model.eval() 
    with torch.no_grad():
           

        X_0_cnn = new_model(image)
        linear = torch.matmul(X_0_cnn,model_linear.t())
        Rel = torch.nn.ReLU() 
        relu = Rel(linear)
        weights = torch.load("ACIL_weights4_new.pth",map_location='cpu')
        print(relu.size(),weights.size())
        output = torch.matmul(relu,weights.t())
    return output

def predict5(image,model):
    model.layer4 = nn.Sequential()
    model.maxpool = nn.Sequential()
    image = preprocess_image(image)


    new_model = torch.nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2,
                                        model.layer3, model.layer4, model.avgpool, Flatten())

    model_linear = torch.load("ACIL_weights_linear_5_new.pth",map_location='cpu')
    #print(summary(new_model,input_size=(3,32,32)))

    new_model.eval() 
    with torch.no_grad():
           

        X_0_cnn = new_model(image)
        linear = torch.matmul(X_0_cnn,model_linear.t())
        Rel = torch.nn.ReLU() 
        relu = Rel(linear)
        weights = torch.load("ACIL_weights5_new.pth",map_location='cpu')
        print(relu.size(),weights.size())
        output = torch.matmul(relu,weights.t())
    return output

def predict6(image,model):
    model.layer4 = nn.Sequential()
    model.maxpool = nn.Sequential()
    image = preprocess_image(image)


    new_model = torch.nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2,
                                        model.layer3, model.layer4, model.avgpool, Flatten())

    model_linear = torch.load("ACIL_weights_linear_6_new.pth",map_location='cpu')
    #print(summary(new_model,input_size=(3,32,32)))

    new_model.eval() 
    with torch.no_grad():
           

        X_0_cnn = new_model(image)
        linear = torch.matmul(X_0_cnn,model_linear.t())
        Rel = torch.nn.ReLU() 
        relu = Rel(linear)
        weights = torch.load("ACIL_weights6_new.pth",map_location='cpu')
        print(relu.size(),weights.size())
        output = torch.matmul(relu,weights.t())
    return output

def predict7(image,model):
    model.layer4 = nn.Sequential()
    model.maxpool = nn.Sequential()
    image = preprocess_image(image)


    new_model = torch.nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2,
                                        model.layer3, model.layer4, model.avgpool, Flatten())

    model_linear = torch.load("ACIL_weights_linear_7_new.pth",map_location='cpu')
    #print(summary(new_model,input_size=(3,32,32)))

    new_model.eval() 
    with torch.no_grad():
           

        X_0_cnn = new_model(image)
        linear = torch.matmul(X_0_cnn,model_linear.t())
        Rel = torch.nn.ReLU() 
        relu = Rel(linear)
        weights = torch.load("ACIL_weights7_new.pth",map_location='cpu')
        print(relu.size(),weights.size())
        output = torch.matmul(relu,weights.t())
    return output

def predict8(image,model):
    model.layer4 = nn.Sequential()
    model.maxpool = nn.Sequential()
    image = preprocess_image(image)


    new_model = torch.nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2,
                                        model.layer3, model.layer4, model.avgpool, Flatten())

    model_linear = torch.load("ACIL_weights_linear_8_new.pth",map_location='cpu')
    #print(summary(new_model,input_size=(3,32,32)))

    new_model.eval() 
    with torch.no_grad():
           

        X_0_cnn = new_model(image)
        linear = torch.matmul(X_0_cnn,model_linear.t())
        Rel = torch.nn.ReLU() 
        relu = Rel(linear)
        weights = torch.load("ACIL_weights8_new.pth",map_location='cpu')
        print(relu.size(),weights.size())
        output = torch.matmul(relu,weights.t())
    return output

def predict9(image,model):
    model.layer4 = nn.Sequential()
    model.maxpool = nn.Sequential()
    image = preprocess_image(image)


    new_model = torch.nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2,
                                        model.layer3, model.layer4, model.avgpool, Flatten())

    model_linear = torch.load("ACIL_weights_linear_9_new.pth",map_location='cpu')
    #print(summary(new_model,input_size=(3,32,32)))

    new_model.eval() 
    with torch.no_grad():
           

        X_0_cnn = new_model(image)
        linear = torch.matmul(X_0_cnn,model_linear.t())
        Rel = torch.nn.ReLU() 
        relu = Rel(linear)
        weights = torch.load("ACIL_weights9_new.pth",map_location='cpu')
        print(relu.size(),weights.size())
        output = torch.matmul(relu,weights.t())
    return output










# Streamlit app
def main():
    st.title("Few Shot Class Incremental Learning")
    st.write("Team Name: ML_Coders , Roll no: 22M2021 & 200040067 ")
    st.write("Upload an image and let the model predict its class.")
    selected_model = st.selectbox("Select Model", ("Base-model","Phase0-model","Phase1-model","Phase2-model","Phase3-model","Phase4-model","Phase5-model","Phase6-model","Phase7-model","Phase8-model","Phase9-model"))
    selected_dataset = st.selectbox("Select Dataset",("CIFAR-100","MINI-IMAGENET","CUB-200"))

    if selected_dataset =="CIFAR-100":
        metadata_path = 'meta.txt' # change this path`\
        metadata = unpickle(metadata_path)
        fine_class_dict = dict(list(enumerate(metadata[b'fine_label_names'])))
        classes = list(fine_class_dict.values())
        print(classes[:50],len(classes[0:50]))
        model = model_cifar.__dict__["resnet32"](50)
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)

        checkpoint = torch.load("save_model/cifar100/resnet32/1-10_9-59-52-757/checkpoint.pth.tar", map_location='cpu')

        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    
        if selected_model == "Base-model":
            st.write("supported classes: ",classes[:50])

            model = model_cifar.__dict__["resnet32"](50)
            optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)

            checkpoint = torch.load("save_model/cifar100/resnet32/1-10_9-59-52-757/checkpoint.pth.tar", map_location='cpu')

            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

        if selected_model == "Phase0-model":
            st.write("supported classes: ",classes[:55])
            st.write("Newly added classes: ",classes[50:55] )

        if selected_model == "Phase1-model":
            st.write("supported classes: ",classes[:60])
            st.write("Newly added classes: ",classes[55:60] )

        if selected_model == "Phase2-model":
            st.write("supported classes: ",classes[:65])
            st.write("Newly added classes: ",classes[60:65] )

        if selected_model == "Phase3-model":
            st.write("supported classes: ",classes[:70])
            st.write("Newly added classes: ",classes[65:70] )

        if selected_model == "Phase4-model":
            st.write("supported classes: ",classes[:75])
            st.write("Newly added classes: ",classes[70:75] )

        if selected_model == "Phase5-model":
            st.write("supported classes: ",classes[:80])
            st.write("Newly added classes: ",classes[75:80] )

        if selected_model == "Phase6-model":
            st.write("supported classes: ",classes[:85])
            st.write("Newly added classes: ",classes[80:85] )

        if selected_model == "Phase7-model":
            st.write("supported classes: ",classes[:90])
            st.write("Newly added classes: ",classes[85:90] )

        if selected_model == "Phase8-model":
            st.write("supported classes: ",classes[:95])
            st.write("Newly added classes: ",classes[90:95] )
        if selected_model == "Phase9-model":
            st.write("supported classes: ",classes[:100])
            st.write("Newly added classes: ",classes[95:100] )

        
        model.layer4 = nn.Sequential()
        model.maxpool = nn.Sequential()
        print("Phase1-model")

    if selected_dataset =="CUB-200":
        import torchvision.models as models
        import pandas as pd
        model = models.__dict__['resnet18'](50)
        CUB = pd.read_csv("CUB_labels.csv")
        optimizer = torch.optim.SGD(model.parameters(), 0.1,
                                    momentum=0.9,
                                    weight_decay=1e-4)

        checkpoint = torch.load("save_model/CUB200/resnet18/5-5_10-59-3-10/checkpoint.pth.tar", map_location='cpu')
        if hasattr(model, 'module'):
            model = model.module
        else:
            model = model
        if hasattr(model, 'layer4'):
                    pass
        else:
                    model.layer4 = nn.Sequential()
        if hasattr(model, 'maxpool'):
                    pass
        else:
                    model.maxpool = nn.Sequential()
        model.eval()
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
       

        if selected_model == "Base-model":
            st.write("supported classes: ",CUB.iloc[:100]['Name'])
            #st.write("Newly added classes: ",CUB.iloc[100:110]['Name'] )

        if selected_model == "Phase0-model":
            st.write("supported classes: ",CUB.iloc[:110]['Name'])
            st.write("Newly added classes: ",CUB.iloc[100:110]['Name'] )

        if selected_model == "Phase1-model":
            st.write("supported classes: ",CUB.iloc[:120]['Name'])
            st.write("Newly added classes: ",CUB.iloc[110:120]['Name'] )

        if selected_model == "Phase2-model":
            st.write("supported classes: ",CUB.iloc[:130]['Name'])
            st.write("Newly added classes: ",CUB.iloc[120:130]['Name'] )

        if selected_model == "Phase3-model":
            st.write("supported classes: ",CUB.iloc[:140]['Name'])
            st.write("Newly added classes: ",CUB.iloc[130:140]['Name'] )

        if selected_model == "Phase4-model":
            st.write("supported classes: ",CUB.iloc[:150]['Name'])
            st.write("Newly added classes: ",CUB.iloc[140:150]['Name'] )

        if selected_model == "Phase5-model":
            st.write("supported classes: ",CUB.iloc[:160]['Name'])
            st.write("Newly added classes: ",CUB.iloc[150:160]['Name'] )

        if selected_model == "Phase6-model":
            st.write("supported classes: ",CUB.iloc[:170]['Name'])
            st.write("Newly added classes: ",CUB.iloc[160:170]['Name'] )

        if selected_model == "Phase7-model":
            st.write("supported classes: ",CUB.iloc[:180]['Name'])
            st.write("Newly added classes: ",CUB.iloc[170:180]['Name'] )

        if selected_model == "Phase8-model":
            st.write("supported classes: ",CUB.iloc[:190]['Name'])
            st.write("Newly added classes: ",CUB.iloc[180:190]['Name'] )
        if selected_model == "Phase9-model":
            st.write("supported classes: ",CUB.iloc[:200]['Name'])
            st.write("Newly added classes: ",CUB.iloc[190:200]['Name'] )

    if selected_dataset =="MINI-IMAGENET":
        import torchvision.models as models
        import pandas as pd
        model = models.__dict__['resnet18'](50)
        Imagenet = pd.read_csv("Mini-imagenet-labels.csv")
        optimizer = torch.optim.SGD(model.parameters(), 0.1,
                                    momentum=0.9,
                                    weight_decay=1e-4)

        checkpoint = torch.load("save_model/CUB200/resnet18/5-5_10-59-3-10/checkpoint.pth.tar", map_location='cpu')
        if hasattr(model, 'module'):
            model = model.module
        else:
            model = model
        if hasattr(model, 'layer4'):
                    pass
        else:
                    model.layer4 = nn.Sequential()
        if hasattr(model, 'maxpool'):
                    pass
        else:
                    model.maxpool = nn.Sequential()
        model.eval()
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
       

        if selected_model == "Base-model":
            st.write("supported classes: ",Imagenet.iloc[:50]['names'])
            #st.write("Newly added classes: ",CUB.iloc[100:110]['Name'] )

        if selected_model == "Phase0-model":
            st.write("supported classes: ",Imagenet.iloc[:55]['names'])
            st.write("Newly added classes: ",Imagenet.iloc[50:55]['names'] )

        if selected_model == "Phase1-model":
            st.write("supported classes: ",Imagenet.iloc[:60]['names'])
            st.write("Newly added classes: ",Imagenet.iloc[55:60]['names'] )

        if selected_model == "Phase2-model":
            st.write("supported classes: ",Imagenet.iloc[:65]['names'])
            st.write("Newly added classes: ",Imagenet.iloc[60:65]['names'] )

        if selected_model == "Phase3-model":
            st.write("supported classes: ",Imagenet.iloc[:70]['names'])
            st.write("Newly added classes: ",Imagenet.iloc[65:70]['names'] )
        if selected_model == "Phase4-model":
            st.write("supported classes: ",Imagenet.iloc[:75]['names'])
            st.write("Newly added classes: ",Imagenet.iloc[70:75]['names'] )

        if selected_model == "Phase5-model":
            st.write("supported classes: ",Imagenet.iloc[:80]['names'])
            st.write("Newly added classes: ",Imagenet.iloc[75:80]['names'] )
        if selected_model == "Phase6-model":
            st.write("supported classes: ",Imagenet.iloc[:85]['names'])
            st.write("Newly added classes: ",Imagenet.iloc[80:85]['names'] )

        if selected_model == "Phase7-model":
            st.write("supported classes: ",Imagenet.iloc[:90]['names'])
            st.write("Newly added classes: ",Imagenet.iloc[85:90]['names'] )

        if selected_model == "Phase8-model":
            st.write("supported classes: ",Imagenet.iloc[:95]['names'])
            st.write("Newly added classes: ",Imagenet.iloc[90:95]['names'] )
        if selected_model == "Phase9-model":
            st.write("supported classes: ",Imagenet.iloc[:100]['names'])
            st.write("Newly added classes: ",Imagenet.iloc[95:100]['names'] )




        

        

    # Upload image
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image',width = 10 , use_column_width=True)

        # Check if a button is clicked to predict
        if st.button('Predict'):
            if selected_dataset =="CIFAR-100":
                if selected_model == "Base-model":
                # Make predictions
                    predictions = predict(image,model)

                    predicted_class = torch.argmax(predictions, dim=1).item()

                    sorted_index = torch.argsort(predictions,descending=True)
                    sorted_values = torch.sort(predictions,descending=True).values
                    top_5_idices = sorted_index[:5]
                    print(sorted_index[0][:5])
                    # Display prediction results
                    
                    for i in sorted_index[0][:5]:
                        print(i.item())
                        print("predicted_class",fine_class_dict[i.item()])
                        st.write("predicted_class",fine_class_dict[i.item()])


                if selected_model == "Phase0-model":
                    predictions = predict0(image,model)
                    print(predictions.size())
                    predicted_class = torch.argmax(predictions, dim=1).item()

                    sorted_index = torch.argsort(predictions,descending=True)
                    print(sorted_index[0][:5])
                    # Display prediction results
                    
                    for i in sorted_index[0][:5]:
                        print(i.item())
                        print("predicted_class",fine_class_dict[i.item()])
                        st.write("predicted_class",fine_class_dict[i.item()])

                if selected_model == "Phase1-model":
                    predictions = predict1(image,model)
                    print(predictions.size())
                    predicted_class = torch.argmax(predictions, dim=1).item()

                    sorted_index = torch.argsort(predictions,descending=True)
                    print(sorted_index[0][:5])
                    # Display prediction results
                    
                    for i in sorted_index[0][:5]:
                        print(i.item())
                        print("predicted_class",fine_class_dict[i.item()])
                        st.write("predicted_class",fine_class_dict[i.item()])

                if selected_model == "Phase2-model":
                    predictions = predict2(image,model)
                    predicted_class = torch.argmax(predictions, dim=1).item()
                    sorted_index = torch.argsort(predictions,descending=True)
                    print(sorted_index[0][:5])
                    # Display prediction results
                    
                    for i in sorted_index[0][:5]:
                        print(i.item())
                        print("predicted_class",fine_class_dict[i.item()])
                        st.write("predicted_class",fine_class_dict[i.item()])


                if selected_model == "Phase3-model":
                    predictions = predict3(image,model)
                    predicted_class = torch.argmax(predictions, dim=1).item()
                    sorted_index = torch.argsort(predictions,descending=True)
                    print(sorted_index[0][:5])
                    # Display prediction results
                    
                    for i in sorted_index[0][:5]:
                        print(i.item())
                        print("predicted_class",fine_class_dict[i.item()])
                        st.write("predicted_class",fine_class_dict[i.item()])
                    #st.write("Final-model")

                if selected_model == "Phase4-model":
                    predictions = predict4(image,model)
                    predicted_class = torch.argmax(predictions, dim=1).item()
                    sorted_index = torch.argsort(predictions,descending=True)
                    print(sorted_index[0][:5])
                    # Display prediction results
                    
                    for i in sorted_index[0][:5]:
                        print(i.item())
                        print("predicted_class",fine_class_dict[i.item()])
                        st.write("predicted_class",fine_class_dict[i.item()])

                if selected_model == "Phase5-model":
                    predictions = predict5(image,model)
                    predicted_class = torch.argmax(predictions, dim=1).item()
                    sorted_index = torch.argsort(predictions,descending=True)
                    print(sorted_index[0][:5])
                    # Display prediction results
                    
                    for i in sorted_index[0][:5]:
                        print(i.item())
                        print("predicted_class",fine_class_dict[i.item()])
                        st.write("predicted_class",fine_class_dict[i.item()])

                if selected_model == "Phase6-model":
                    predictions = predict6(image,model)
                    predicted_class = torch.argmax(predictions, dim=1).item()
                    sorted_index = torch.argsort(predictions,descending=True)
                    print(sorted_index[0][:5])
                    # Display prediction results
                    
                    for i in sorted_index[0][:5]:
                        print(i.item())
                        print("predicted_class",fine_class_dict[i.item()])
                        st.write("predicted_class",fine_class_dict[i.item()])

                if selected_model == "Phase7-model":
                    predictions = predict7(image,model)
                    predicted_class = torch.argmax(predictions, dim=1).item()
                    sorted_index = torch.argsort(predictions,descending=True)
                    print(sorted_index[0][:5])
                    # Display prediction results
                    
                    for i in sorted_index[0][:5]:
                        print(i.item())
                        print("predicted_class",fine_class_dict[i.item()])
                        st.write("predicted_class",fine_class_dict[i.item()])

                if selected_model == "Phase8-model":
                    predictions = predict8(image,model)
                    predicted_class = torch.argmax(predictions, dim=1).item()
                    sorted_index = torch.argsort(predictions,descending=True)
                    print(sorted_index[0][:5])
                    # Display prediction results
                    
                    for i in sorted_index[0][:5]:
                        print(i.item())
                        print("predicted_class",fine_class_dict[i.item()])
                        st.write("predicted_class",fine_class_dict[i.item()])

                if selected_model == "Phase9-model":
                    predictions = predict9(image,model)
                    predicted_class = torch.argmax(predictions, dim=1).item()
                    sorted_index = torch.argsort(predictions,descending=True)
                    print(sorted_index[0][:5])
                    # Display prediction results
                    
                    for i in sorted_index[0][:5]:
                        print(i.item())
                        print("predicted_class",fine_class_dict[i.item()])
                        st.write("predicted_class",fine_class_dict[i.item()])

            if selected_dataset =="CUB-200":
                if selected_model == "Base-model":
                # Make predictions
                    predictions = CUBpredict0(image,model)

                    predicted_class = torch.argmax(predictions, dim=1).item()

                    sorted_index = torch.argsort(predictions,descending=True)
                    #sorted_values = torch.sort(predictions,descending=True).values
                    #top_5_idices = sorted_index[:5]
                    print(sorted_index[0][:5])
                    # Display prediction results
                    
                    for i in sorted_index[0][:5]:
                        print(i.item())
                        print("predicted_class",CUB.iloc[int(i)]['Name'])
                        st.write("predicted_class",CUB.iloc[int(i)]['Name'])
                if selected_model == "Phase0-model":
                # Make predictions
                    predictions = CUBpredict0(image,model)

                    predicted_class = torch.argmax(predictions, dim=1).item()

                    sorted_index = torch.argsort(predictions,descending=True)
                    #sorted_values = torch.sort(predictions,descending=True).values
                    #top_5_idices = sorted_index[:5]
                    print(sorted_index[0][:5])
                    # Display prediction results
                    
                    for i in sorted_index[0][:5]:
                        print(i.item())
                        print("predicted_class",CUB.iloc[int(i)]['Name'])
                        st.write("predicted_class",CUB.iloc[int(i)]['Name'])

                if selected_model == "Phase1-model":
                # Make predictions
                    predictions = CUBpredict1(image,model)

                    predicted_class = torch.argmax(predictions, dim=1).item()

                    sorted_index = torch.argsort(predictions,descending=True)
                    #sorted_values = torch.sort(predictions,descending=True).values
                    #top_5_idices = sorted_index[:5]
                    print(sorted_index[0][:5])
                    # Display prediction results
                    
                    for i in sorted_index[0][:5]:
                        print(i.item())
                        print("predicted_class",CUB.iloc[int(i)]['Name'])
                        st.write("predicted_class",CUB.iloc[int(i)]['Name'])

                if selected_model == "Phase2-model":
                # Make predictions
                    predictions = CUBpredict2(image,model)

                    predicted_class = torch.argmax(predictions, dim=1).item()

                    sorted_index = torch.argsort(predictions,descending=True)
                    #sorted_values = torch.sort(predictions,descending=True).values
                    #top_5_idices = sorted_index[:5]
                    print(sorted_index[0][:5])
                    # Display prediction results
                    
                    for i in sorted_index[0][:5]:
                        print(i.item())
                        print("predicted_class",CUB.iloc[int(i)]['Name'])
                        st.write("predicted_class",CUB.iloc[int(i)]['Name'])

                if selected_model == "Phase3-model":
                # Make predictions
                    predictions = CUBpredict3(image,model)

                    predicted_class = torch.argmax(predictions, dim=1).item()

                    sorted_index = torch.argsort(predictions,descending=True)
                    #sorted_values = torch.sort(predictions,descending=True).values
                    #top_5_idices = sorted_index[:5]
                    print(sorted_index[0][:5])
                    # Display prediction results
                    
                    for i in sorted_index[0][:5]:
                        print(i.item())
                        print("predicted_class",CUB.iloc[int(i)]['Name'])
                        st.write("predicted_class",CUB.iloc[int(i)]['Name'])

                if selected_model == "Phase4-model":
                # Make predictions
                    predictions = CUBpredict4(image,model)

                    predicted_class = torch.argmax(predictions, dim=1).item()

                    sorted_index = torch.argsort(predictions,descending=True)
                    #sorted_values = torch.sort(predictions,descending=True).values
                    #top_5_idices = sorted_index[:5]
                    print(sorted_index[0][:5])
                    # Display prediction results
                    
                    for i in sorted_index[0][:5]:
                        print(i.item())
                        print("predicted_class",CUB.iloc[int(i)]['Name'])
                        st.write("predicted_class",CUB.iloc[int(i)]['Name'])

                if selected_model == "Phase5-model":
                # Make predictions
                    predictions = CUBpredict5(image,model)

                    predicted_class = torch.argmax(predictions, dim=1).item()

                    sorted_index = torch.argsort(predictions,descending=True)
                    #sorted_values = torch.sort(predictions,descending=True).values
                    #top_5_idices = sorted_index[:5]
                    print(sorted_index[0][:5])
                    # Display prediction results
                    
                    for i in sorted_index[0][:5]:
                        print(i.item())
                        print("predicted_class",CUB.iloc[int(i)]['Name'])
                        st.write("predicted_class",CUB.iloc[int(i)]['Name'])

                if selected_model == "Phase6-model":
                # Make predictions
                    predictions = CUBpredict6(image,model)

                    predicted_class = torch.argmax(predictions, dim=1).item()

                    sorted_index = torch.argsort(predictions,descending=True)
                    #sorted_values = torch.sort(predictions,descending=True).values
                    #top_5_idices = sorted_index[:5]
                    print(sorted_index[0][:5])
                    # Display prediction results
                    
                    for i in sorted_index[0][:5]:
                        print(i.item())
                        print("predicted_class",CUB.iloc[int(i)]['Name'])
                        st.write("predicted_class",CUB.iloc[int(i)]['Name'])

                if selected_model == "Phase7-model":
                # Make predictions
                    predictions = CUBpredict7(image,model)

                    predicted_class = torch.argmax(predictions, dim=1).item()

                    sorted_index = torch.argsort(predictions,descending=True)
                    #sorted_values = torch.sort(predictions,descending=True).values
                    #top_5_idices = sorted_index[:5]
                    print(sorted_index[0][:5])
                    # Display prediction results
                    
                    for i in sorted_index[0][:5]:
                        print(i.item())
                        print("predicted_class",CUB.iloc[int(i)]['Name'])
                        st.write("predicted_class",CUB.iloc[int(i)]['Name'])

                if selected_model == "Phase8-model":
                # Make predictions
                    predictions = CUBpredict8(image,model)

                    predicted_class = torch.argmax(predictions, dim=1).item()

                    sorted_index = torch.argsort(predictions,descending=True)
                    #sorted_values = torch.sort(predictions,descending=True).values
                    #top_5_idices = sorted_index[:5]
                    print(sorted_index[0][:5])
                    # Display prediction results
                    
                    for i in sorted_index[0][:5]:
                        print(i.item())
                        print("predicted_class",CUB.iloc[int(i)]['Name'])
                        st.write("predicted_class",CUB.iloc[int(i)]['Name'])
                if selected_model == "Phase9-model":
                # Make predictions
                    predictions = CUBpredict9(image,model)

                    predicted_class = torch.argmax(predictions, dim=1).item()

                    sorted_index = torch.argsort(predictions,descending=True)
                    #sorted_values = torch.sort(predictions,descending=True).values
                    #top_5_idices = sorted_index[:5]
                    print(sorted_index[0][:5])
                    # Display prediction results
                    
                    for i in sorted_index[0][:5]:
                        print(i.item())
                        print("predicted_class",CUB.iloc[int(i)]['Name'])
                        st.write("predicted_class",CUB.iloc[int(i)]['Name'])

            if selected_dataset =="MINI-IMAGENET":
                if selected_model == "Base-model":
                # Make predictions
                    predictions = minipredict0(image,model)

                    predicted_class = torch.argmax(predictions, dim=1).item()

                    sorted_index = torch.argsort(predictions,descending=True)
                    #sorted_values = torch.sort(predictions,descending=True).values
                    #top_5_idices = sorted_index[:5]
                    print(sorted_index[0][:5])
                    # Display prediction results
                    
                    for i in sorted_index[0][:5]:
                        print(i.item())
                        print("predicted_class",Imagenet.iloc[int(i)]['names'])
                        st.write("predicted_class",Imagenet.iloc[int(i)]['names'])

                if selected_model == "Phase0-model":
                # Make predictions
                    predictions = minipredict0(image,model)

                    predicted_class = torch.argmax(predictions, dim=1).item()

                    sorted_index = torch.argsort(predictions,descending=True)
                    #sorted_values = torch.sort(predictions,descending=True).values
                    #top_5_idices = sorted_index[:5]
                    print(sorted_index[0][:5])
                    # Display prediction results
                    
                    for i in sorted_index[0][:5]:
                        print(i.item())
                        print("predicted_class",Imagenet.iloc[int(i)]['names'])
                        st.write("predicted_class",Imagenet.iloc[int(i)]['names'])

                if selected_model == "Phase1-model":
                # Make predictions
                    predictions = minipredict1(image,model)

                    predicted_class = torch.argmax(predictions, dim=1).item()

                    sorted_index = torch.argsort(predictions,descending=True)
                    #sorted_values = torch.sort(predictions,descending=True).values
                    #top_5_idices = sorted_index[:5]
                    print(sorted_index[0][:5])
                    # Display prediction results
                    
                    for i in sorted_index[0][:5]:
                        print(i.item())
                        print("predicted_class",Imagenet.iloc[int(i)]['names'])
                        st.write("predicted_class",Imagenet.iloc[int(i)]['names'])

                if selected_model == "Phase2-model":
                # Make predictions
                    predictions = minipredict2(image,model)

                    predicted_class = torch.argmax(predictions, dim=1).item()

                    sorted_index = torch.argsort(predictions,descending=True)
                    #sorted_values = torch.sort(predictions,descending=True).values
                    #top_5_idices = sorted_index[:5]
                    print(sorted_index[0][:5])
                    # Display prediction results
                    
                    for i in sorted_index[0][:5]:
                        print(i.item())
                        print("predicted_class",Imagenet.iloc[int(i)]['names'])
                        st.write("predicted_class",Imagenet.iloc[int(i)]['names'])

                if selected_model == "Phase3-model":
                # Make predictions
                    predictions = minipredict3(image,model)

                    predicted_class = torch.argmax(predictions, dim=1).item()

                    sorted_index = torch.argsort(predictions,descending=True)
                    #sorted_values = torch.sort(predictions,descending=True).values
                    #top_5_idices = sorted_index[:5]
                    print(sorted_index[0][:5])
                    # Display prediction results
                    
                    for i in sorted_index[0][:5]:
                        print(i.item())
                        print("predicted_class",Imagenet.iloc[int(i)]['names'])
                        st.write("predicted_class",Imagenet.iloc[int(i)]['names'])

                if selected_model == "Phase4-model":
                # Make predictions
                    predictions = minipredict4(image,model)

                    predicted_class = torch.argmax(predictions, dim=1).item()

                    sorted_index = torch.argsort(predictions,descending=True)
                    #sorted_values = torch.sort(predictions,descending=True).values
                    #top_5_idices = sorted_index[:5]
                    print(sorted_index[0][:5])
                    # Display prediction results
                    
                    for i in sorted_index[0][:5]:
                        print(i.item())
                        print("predicted_class",Imagenet.iloc[int(i)]['names'])
                        st.write("predicted_class",Imagenet.iloc[int(i)]['names'])

                if selected_model == "Phase5-model":
                # Make predictions
                    predictions = minipredict5(image,model)

                    predicted_class = torch.argmax(predictions, dim=1).item()

                    sorted_index = torch.argsort(predictions,descending=True)
                    #sorted_values = torch.sort(predictions,descending=True).values
                    #top_5_idices = sorted_index[:5]
                    print(sorted_index[0][:5])
                    # Display prediction results
                    
                    for i in sorted_index[0][:5]:
                        print(i.item())
                        print("predicted_class",Imagenet.iloc[int(i)]['names'])
                        st.write("predicted_class",Imagenet.iloc[int(i)]['names'])

                if selected_model == "Phase6-model":
                # Make predictions
                    predictions = minipredict6(image,model)

                    predicted_class = torch.argmax(predictions, dim=1).item()

                    sorted_index = torch.argsort(predictions,descending=True)
                    #sorted_values = torch.sort(predictions,descending=True).values
                    #top_5_idices = sorted_index[:5]
                    print(sorted_index[0][:5])
                    # Display prediction results
                    
                    for i in sorted_index[0][:5]:
                        print(i.item())
                        print("predicted_class",Imagenet.iloc[int(i)]['names'])
                        st.write("predicted_class",Imagenet.iloc[int(i)]['names'])

                if selected_model == "Phase7-model":
                # Make predictions
                    predictions = minipredict7(image,model)

                    predicted_class = torch.argmax(predictions, dim=1).item()

                    sorted_index = torch.argsort(predictions,descending=True)
                    #sorted_values = torch.sort(predictions,descending=True).values
                    #top_5_idices = sorted_index[:5]
                    print(sorted_index[0][:5])
                    # Display prediction results
                    
                    for i in sorted_index[0][:5]:
                        print(i.item())
                        print("predicted_class",Imagenet.iloc[int(i)]['names'])
                        st.write("predicted_class",Imagenet.iloc[int(i)]['names'])

                if selected_model == "Phase8-model":
                # Make predictions
                    predictions = minipredict8(image,model)

                    predicted_class = torch.argmax(predictions, dim=1).item()

                    sorted_index = torch.argsort(predictions,descending=True)
                    #sorted_values = torch.sort(predictions,descending=True).values
                    #top_5_idices = sorted_index[:5]
                    print(sorted_index[0][:5])
                    # Display prediction results
                    
                    for i in sorted_index[0][:5]:
                        print(i.item())
                        print("predicted_class",Imagenet.iloc[int(i)]['names'])
                        st.write("predicted_class",Imagenet.iloc[int(i)]['names'])

                if selected_model == "Phase9-model":
                # Make predictions
                    predictions = minipredict9(image,model)

                    predicted_class = torch.argmax(predictions, dim=1).item()

                    sorted_index = torch.argsort(predictions,descending=True)
                    #sorted_values = torch.sort(predictions,descending=True).values
                    #top_5_idices = sorted_index[:5]
                    print(sorted_index[0][:5])
                    # Display prediction results
                    
                    for i in sorted_index[0][:5]:
                        print(i.item())
                        print("predicted_class",Imagenet.iloc[int(i)]['names'])
                        st.write("predicted_class",Imagenet.iloc[int(i)]['names'])



            

# Run the app
if __name__ == '__main__':
    main()
