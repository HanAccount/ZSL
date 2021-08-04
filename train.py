import numpy as np
import torch
from torch import nn
from AnimalDataset import AnimalDataset
import torchvision
import torchvision.transforms as transform
from torch.utils.data import DataLoader
import os
import sys
import argparse
from tqdm import tqdm

# 建立模型
# 使用的是resnet50模型
def build_model(num_labels, is_pretrained, is_parallel,):
    """

    :param num_labels: 输出的维度
    :param is_pretrained: 是否使用预训练模型
    :param is_parallel: 是否使用多gpu

    """
    model = torchvision.models.resnet50(pretrained=is_pretrained).to(device)
    if is_pretrained:
        for i , param in model.named_parameters():
            param.requires_grad = False
    if is_parallel:
        print('Using DataParallel')
        model = nn.DataParallel(model)
        # 最后fc层的输入
        model_features = model.module.fc.in_features
        model.fc = nn.Sequential(nn.BatchNorm1d(model_features), nn.ReLU(), nn.Dropout(0.25),
                                 nn.Linear(model_features,num_labels))
    else:
        print('Not Using DataParallel')
        model_features = model.fc.in_features
        model.fc = nn.Sequential(nn.BatchNorm1d(model_features), nn.ReLU(), nn.Dropout(0.25),
                                 nn.Linear(model_features, num_labels))

    return model

# 训练
def train(num_epochs, eval_interval, learning_rate, output_filename, model_name, optimizer_name, batch_size, device):
    train_params = {'batch_size':batch_size, 'shuffle':True, 'num_workers':0}
    test_params = {'batch_size':1, 'shuffle':True, 'num_workers':0}

    # transform
    train_process_steps = transform.Compose([
        transform.RandomRotation(15),
        transform.RandomHorizontalFlip(),
        transform.ColorJitter(brightness=0.3, contrast=0.3),
        transform.Resize((224,224)),
        transform.ToTensor()
    ])
    test_process_steps = transform.Compose([
        transform.Resize((224,224)),
        transform.ToTensor()
    ])

    # dataset
    train_dataset = AnimalDataset('trainclasses.txt', transform=train_process_steps)
    test_dataset = AnimalDataset('testclasses.txt', transform=test_process_steps)

    # DataLoader
    train_loader = DataLoader(train_dataset, **train_params)
    test_loader = DataLoader(test_dataset, **test_params)

    # LOSS
    criterion = nn.BCELoss()

    total_steps = len(train_loader)

    # 多gpu运行
    if torch.cuda.device_count() > 1:
        model = build_model(num_labels, False, True).to(device)
    else:
        model = build_model(num_labels, False, False).to(device)

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 开始训练
    for epoch in range(num_epochs):
        for i, (images, features, img_names, indexs) in enumerate(tqdm(train_loader)):
            if images.shape[0] < 2:
                break
            images = images.to(device)
            features = features.to(device).float()

            model.train()
            # forward
            outputs = torch.sigmoid(model(images))
            loss = criterion(outputs, features)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 50 == 0:
                curr_iter = epoch * len(train_loader) + 1
                print('Epoch [{} / {}], step[{} / {}], Batch Loss: {:.4f}'.format(
                    epoch + 1, num_epochs, i + 1, total_steps, loss.item()
                ))

        # evaluations
        if (epoch + 1) % eval_interval == 0:
            print('Evaluating:')
            curr_acc = evaluate(model,test_loader)
            print('Epoch [{} / {}] Approx. training accuracy: {}'.format(
                epoch + 1, num_epochs, curr_acc
            ))

        # Make final predictions
        print('Making predictions:')
        if not os.path.exists('models'):
            os.mkdir('models')
        torch.save(model.state_dict(),'models/{}'.format(model_name))
        torch.save(optimizer.state_dict(),'models/{}'.format(optimizer_name))
        make_predictions(model,test_loader,output_filename)

def evaluate(model, dataloader):
    model.eval()

    mean_acc = 0.0
    pred_classes = []
    truth_classes = []

    with torch.no_grad():
        for i , (images, features, img_names, indexs) in enumerate(tqdm(dataloader)):
            images, features = images.to(device),features.to(device).float()

            outputs = model(images)

            pred_labels = torch.sigmoid(outputs)
            curr_pred_classes = labels_to_class(pred_labels)
            pred_classes.extend(curr_pred_classes)

            curr_truth_classes = []
            for index in indexs:
                curr_truth_classes.append(classes[index])
            truth_classes.extend(curr_truth_classes)

    pred_classes = np.array(pred_classes)
    truth_classes = np.array(truth_classes)
    mean_acc = np.mean(pred_classes == truth_classes)

    # Reset
    model.train()
    return mean_acc
# 欧几里得距离
def get_euclidean_dist(curr_labels, class_labels):
    return np.sqrt(np.sum((curr_labels - class_labels) ** 2))

# 标签到类的映射
def labels_to_class(pred_labels):
    # predicate_binary_mat = np.array(np.genfromtxt(path + 'predicate-matrix-binary.txt', dtype=int))
    predictions = []
    for i in range(pred_labels.shape[0]):
        curr_labels = pred_labels[i,:].cpu().detach().numpy()
        best_dict = sys.maxsize
        best_index = -1
        for j in range(predicate_binary_mat.shape[0]):
            class_labels = predicate_binary_mat[j,:]
            # 预测的标签跟每一个真实标签计算欧几里得距离
            dist = get_euclidean_dist(curr_labels, class_labels)

            if dist < best_dict and classes[j] not in train_classes:
                best_index = j
                best_dict = dist
        predictions.append(classes[best_index])

    return predictions

def make_predictions(model, dataloader, output_filename):
    model.eval()

    pred_classes = []
    output_img_names = []
    with torch.no_grad():
        for i, (images, features, img_names, indexs) in enumerate(tqdm(dataloader)):
            images,features = images.to(device),features.to(device).float()
            outputs = model(images)

            pred_labels = torch.sigmoid(outputs)

            curr_pred_classes = labels_to_class(pred_labels)

            pred_classes.extend(curr_pred_classes)
            output_img_names.extend(img_names)

            if i % 1000 == 0:
                print('Prediction iter: {}'.format(i))
        with open(output_filename, 'w') as f:
            for i in range(len(pred_classes)):
                output_name = output_img_names[i].replace('../AWA2_Data/AwA2-data/Animals_with_Attributes2/JPEGImages/','')
                f.write(output_name + ' ' + pred_classes[i] + '\n')

def load_model(model_file):
    is_parallel = False
    model = build_model(num_labels,False,is_parallel).to(device)

    if is_parallel:
        model = torch.nn.DataParallel(model)
        dict = torch.load(model_file)
        model = model.module
        model.load_state_dict(dict)
    else:
        state_dict = torch.load(model_file)
        model.load_state_dict(state_dict)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs','-n',type=int, default=25)
    parser.add_argument('--eval_interval','-et',type=int, default=5)
    parser.add_argument('--learning_rate','-lr',type=float, default=0.00001)
    parser.add_argument('--model_name','-mn',type=str, default='model.bin')
    parser.add_argument('--optimizer_name','-opt',type=str, default='optimizer.bin')
    parser.add_argument('--output_file','-o', type=str, default='predictions.txt')
    parser.add_argument('--batch_size','-bs',type=int, default=24)

    args = parser.parse_args()
    args = vars(args)

    num_epochs = args['num_epochs']
    eval_interval = args['eval_interval']
    learning_rate = args['learning_rate']
    model_name = args['model_name']
    optimizer_name = args['optimizer_name']
    output_filename = args['output_file']
    batch_size = args['batch_size']

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    path = '../AWA2_Data/AwA2-data/Animals_with_Attributes2/'
    train_classes = np.array(np.genfromtxt(path + 'trainclasses.txt',dtype=str))
    classes = np.array(np.genfromtxt(path + 'classes.txt',dtype=str))[:,-1]
    predicates = np.array(np.genfromtxt(path + 'predicates.txt',dtype=str))[:,-1]
    predicate_binary_mat = np.array(np.genfromtxt(path + 'predicate-matrix-binary.txt',dtype=int))
    predicate_continuous_mat = np.array(np.genfromtxt(path + 'predicate-matrix-continuous.txt',dtype=float))

    num_labels = len(predicates)
    train(num_epochs,eval_interval,learning_rate,output_filename,model_name,optimizer_name,batch_size,device)
