from DenseUNET import Tiramisu
from DataLoaders import train_loader, validation_loader
from torch.utils.tensorboard import SummaryWriter

import torch.optim as optim
from statistics import mean

model = Tiramisu(3, 6)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def dice_loss(pred, target_mask, eps=1e-6):
    axes = (2, 3)
    numerator = 2 * torch.sum(pred * target_mask, axes)
    denominator = torch.sum(torch.square(pred) + torch.square(target_mask), axes)
    return 1 - torch.mean(torch.div(numerator, denominator))


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


model.apply(init_weights)

check_path = 'check_dense_80.pth'
writer = SummaryWriter("Dense-Net-80")
# check_data = torch.load(check_path)
model.to(device)
# model.load_state_dict(check_data['model_state_dict'])

min_lr = 0.5e-6
max_lr = 0.6

params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(params, lr=max_lr)
# optimizer.load_state_dict(check_data['optimzer_state_dict'])

lr_scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=min_lr, max_lr=max_lr, step_size_up=4 * 67)
# lr_scheduler.load_state_dict(check_data['scheduler_state_dict'])


for epoch in range(80):
    model.train()
    for i, (image, target, filen) in enumerate(train_loader):
        optimizer.zero_grad()
        print(i)
        image = image.to(device)
        target = target.to(device)
        output = model(image)

        loss = dice_loss(output, target)

        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        writer.add_scalar('Train/Loss', loss.item(), 67 * epoch + i)

    print('Loss:' + str(loss) + ' at ' + str(epoch))
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimzer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': lr_scheduler.state_dict()
    },
        check_path
    )

    loss_final = []
    for i, (image, target, filen) in enumerate(validation_loader):
        model.eval()
        with torch.no_grad():
            image = image.to(device)
            target = target.to(device)
            output = model(image)
            loss = dice_loss(output, target)
            loss_final.append(loss.item())
    writer.add_scalar('Test/Loss', mean(loss_final), epoch)
