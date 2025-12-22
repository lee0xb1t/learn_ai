import joblib
import torch
import torch.nn as nn
import pandas as pd


if __name__ == '__main__':
    # data = [16.6,'Yes','No','No',3.0,30.0,'No','Female','55-59','White','Yes','Yes',5.0,'Yes','No','Yes']
    raw = pd.read_csv('../dataset/心脏病数据集/heart_2020_test.csv')
    raw.drop(['GenHealth'], axis=1, inplace=True)

    inputs = raw.iloc[:, 1:]
    labels = raw['HeartDisease']

    # 特征工程
    binary_features = ['Smoking', 'Stroke', 'DiffWalking', 'Sex',
                       'AgeCategory', 'Race', 'PhysicalActivity',
                       'Asthma', 'KidneyDisease', 'SkinCancer', 'Sex']

    label_features = ['AgeCategory', 'Race', 'Diabetic']

    numeric_features = ['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime']

    transformer = joblib.load('../dataset/心脏病数据集/transformer.pkl')
    label_encoder = joblib.load('../dataset/心脏病数据集/label_encoder.pkl')

    # 将数据划分为测试机和训练集
    X_test = transformer.transform(inputs)
    y_test = label_encoder.transform(labels)

    X_test = torch.tensor(X_test, dtype=torch.float32)

    loss_func = nn.CrossEntropyLoss()

    model = torch.load('../dataset/心脏病数据集/heart_2020.pt', weights_only=False)

    model.eval()
    with torch.no_grad():
        test_logits = model(X_test)
        test_preds = test_logits.argmax(dim=1)
        test_labels = label_encoder.inverse_transform(test_preds)
        test_accuracy = (test_preds == y_test).float().mean()
        test_probs = test_logits.softmax(dim=1)
        print("Probe: ", test_probs.sum(dim=1))
        print("Label: ", test_labels)
        print("Accuracy: ", test_accuracy.item())


