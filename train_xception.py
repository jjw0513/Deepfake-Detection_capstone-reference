import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
from torch.optim import lr_scheduler
import argparse
import os
import wandb
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from collections import Counter
from torchvision import transforms

from network.models import model_selection
from dataset.mydataset import MyDataset

#TODO Data Augmetntation FOR East real/fake
xception_default_data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((299, 299)),  # Xception model input
        transforms.RandomHorizontalFlip(),  # 랜덤 좌우 반전
        transforms.RandomRotation(degrees=10),  # 랜덤 회전
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 색상 조절
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 랜덤 이동
        transforms.RandomResizedCrop(size=(299, 299), scale=(0.8, 1.0)),  # 랜덤 크롭 및 리사이즈
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'val': transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
}

def calculate_class_weights(dataset):
    labels = [label for _, label in dataset]
    class_counts = Counter(labels)
    total_count = len(labels)
    class_weights = [total_count / class_counts[i] for i in range(len(class_counts))]
    return torch.FloatTensor(class_weights).cuda()

def main():
    # 하이퍼파라미터 설정
    args = parse.parse_args()
    name = args.name
    train_list = args.train_list
    epoches = args.epoches
    batch_size = args.batch_size
    model_name = args.model_name
    dropout = args.dropout
    learning_rate = args.learning_rate
    step_size = args.step_size
    gamma = args.gamma
    k_folds = args.k_folds
    continue_train = args.continue_train
    model_path = args.model_path
    output_path = os.path.join('./output', name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    wandb.init(project="Deepfake-Xception", name='train_Xception_with_kfold_c0+east')
    wandb.config.update(args)

    full_dataset = MyDataset(txt_path=train_list, transform=xception_default_data_transforms['train'])

    #TODO Calculate Class Weights
    class_weights = calculate_class_weights(full_dataset)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    best_val_acc = 0.0

    for fold, (train_idx, val_idx) in enumerate(kfold.split(full_dataset)):
        print(f'FOLD {fold + 1}/{k_folds}')
        print('-' * 20)

        # 학습 및 검증 데이터셋 생성
        train_subsampler = Subset(full_dataset, train_idx)
        val_subsampler = Subset(full_dataset, val_idx)

        train_loader = DataLoader(train_subsampler, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_subsampler, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

        # 모델 설정 및 가중치 로드
        model = model_selection(modelname='xception', num_out_classes=2, dropout=dropout).cuda()
        model = nn.DataParallel(model)
        if continue_train and os.path.exists(model_path):
            print(f"Loading pre-trained weights from {model_path}")
            try:
                model.module.load_state_dict(torch.load(model_path), strict=True)
                print("Model weights loaded successfully.")
            except RuntimeError as e:
                print(f"Error loading model weights: {e}")
                print("Continuing with random initialization.")
        else:
            if continue_train:
                print(f"Warning: Model weights file not found at {model_path}. Training from scratch.")

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        best_model_wts = model.state_dict()
        for epoch in range(epoches):
            print(f'Epoch {epoch + 1}/{epoches}')
            print('-' * 10)

            model.train()
            train_loss, train_corrects = 0.0, 0.0
            all_labels, all_preds = [], []

            for inputs, labels in train_loader:
                inputs, labels = inputs.cuda(), labels.cuda()
                optimizer.zero_grad()
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_corrects += torch.sum(preds == labels).item()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

            train_loss /= len(train_subsampler)
            train_acc = train_corrects / len(train_subsampler)
            train_f1 = f1_score(all_labels, all_preds, average='macro')

            print(f'Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}')
            wandb.log({"Train Loss": train_loss, "Train Acc": train_acc, "Train F1": train_f1, "Epoch": epoch + 1})

            # Validation phase
            model.eval()
            val_loss, val_corrects = 0.0, 0.0
            all_val_labels, all_val_preds = [], []

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.cuda(), labels.cuda()
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    val_corrects += torch.sum(preds == labels).item()
                    all_val_labels.extend(labels.cpu().numpy())
                    all_val_preds.extend(preds.cpu().numpy())

            val_loss /= len(val_subsampler)
            val_acc = val_corrects / len(val_subsampler)
            val_f1 = f1_score(all_val_labels, all_val_preds, average='macro')

            print(f'Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}')
            wandb.log({"Val Loss": val_loss, "Val Acc": val_acc, "Val F1": val_f1, "Epoch": epoch + 1})

            # Save best model in the current fold
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_wts = model.state_dict()

            scheduler.step()

        # Load the best model weights for this fold
        model.load_state_dict(best_model_wts)
        # `module.` 접두어를 포함한 상태로 저장
        torch.save(model.module.state_dict(), os.path.join(output_path, f"best_model_fold_{fold + 1}.pkl"))
        print(f'Saved best model of fold {fold + 1} with accuracy: {best_val_acc:.4f}')

    print(f'Best Validation Accuracy across all folds: {best_val_acc:.4f}')
    wandb.finish()

if __name__ == '__main__':
    parse = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parse.add_argument('--name', '-n', type=str, default='deepfake_xception_kfold_2')
    parse.add_argument('--train_list', '-tl', type=str, default='./last_labels.txt')
    parse.add_argument('--batch_size', '-bz', type=int, default=64)
    parse.add_argument('--epoches', '-e', type=int, default=30)
    parse.add_argument('--model_name', '-mn', type=str, default='xception_model.pkl')
    parse.add_argument('--dropout', type=float, default=0.5, help="Dropout rate")
    parse.add_argument('--learning_rate', '-lr', type=float, default=0.0001, help="Learning rate for optimizer")
#    parse.add_argument('--learning_rate', '-lr', type=float, default=0.001, help="Learning rate for optimizer")
    parse.add_argument('--step_size', type=int, default=5, help="Step size for LR scheduler")
    parse.add_argument('--gamma', type=float, default=0.5, help="Gamma value for LR scheduler")
    parse.add_argument('--k_folds', type=int, default=5, help="Number of folds for KFold cross-validation")
    parse.add_argument('--continue_train', type=bool, default=True, help="Whether to continue training from saved model weights")
    parse.add_argument('--model_path', '-mp', type=str, default='C:\Users\hail\Deepfake-Detection_capstone-reference\output\deepfake_xception_kfold_2\best_model_fold_1.pkl', help="Path to model weights to load if continuing training")
    main()
