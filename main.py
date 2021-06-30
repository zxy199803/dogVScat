from dataset import Dataset
from config import Config
from Model.AlexNet import AlexNet
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import torch
from utils import time_count, write_csv
from tqdm import tqdm  # 不要使用import tqdm
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import time


my_dataset = Dataset()
my_config = Config()
model = AlexNet()

loss_fn = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=my_config.lr)
writer = SummaryWriter(log_dir=my_config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))

if torch.cuda.is_available():
    model = model.cuda()


@time_count
def train():
    total_batch = 0  # 记录总的batch数，
    for epoch in range(1, my_config.epoch + 1):
        print('Epoch{}/{}'.format(epoch, my_config.epoch))
        print('-' * 40)

        for phase in ['train', 'valid']:
            if phase == 'train':
                print('Training...')
                model.train(True)  # 打开训练模式
            else:
                print('Validing...')
                model.train(False)  # 关闭训练模式

            running_loss = 0.0
            running_correct = 0  # 预测正确的样本个数
            for batch, data in enumerate(my_dataset.data_images_loader[phase], 1):
                X, y = data  # 样本和标签
                X, y = X.cuda(), y.cuda()
                outputs = model(X)
                _, y_pred = torch.max(outputs.detach(), 1)
                optimizer.zero_grad()  # 将Varibale的梯度清零
                loss = loss_fn(outputs, y)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()  # 更新所有参数
                running_loss += loss.detach().item()
                running_correct += torch.sum(y_pred == y)
                if batch % 500 == 0 and phase == 'train':
                    print(
                        'Batch {}/{},Train Loss:{:.2f},Train Acc:{:.2f}%'
                            .format(
                            batch, len(my_dataset.data_images[phase]) / my_config.batch_size, running_loss / batch,
                                   100 * running_correct.item() / (my_config.batch_size * batch)
                        )
                    )
                if batch % 20 == 0 and phase == 'train':
                    writer.add_scalar("loss/train", running_loss / batch, total_batch)
                    writer.add_scalar("acc/train", running_correct.item() / (my_config.batch_size * batch), total_batch)
                if phase == 'train':
                    total_batch += 1
            epoch_loss = running_loss * my_config.batch_size / len(my_dataset.data_images[phase])
            epoch_acc = 100 * running_correct.item() / len(my_dataset.data_images[phase])
            print('{} Loss:{:.2f} Acc:{:.2f}%'.format(phase, epoch_loss, epoch_acc))
            if phase == 'valid':
                writer.add_scalar("loss/val", epoch_loss, total_batch)
                writer.add_scalar("acc/val", epoch_acc, total_batch)


def test():
    data = Dataset(train=False)
    data_loader_test = torch.utils.data.DataLoader(data, batch_size=my_config.batch_size, shuffle=False)

    results = []
    for imgs, path in tqdm(data_loader_test):
        X = imgs.cuda()
        outputs = model(X)
        prob, pred = torch.max(F.softmax(outputs, dim=1).detach(), dim=1)
        batch_results = [(_path.item(), 'dog' if _pred.item() else 'cat')
                         for _path, _pred in zip(path, pred)]
        results += batch_results
    write_csv(results, my_config.result_path)


if __name__ == '__main__':
    train()
    torch.save(model, 'Alexnet.pt')
    test()
