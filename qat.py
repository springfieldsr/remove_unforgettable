from utils import *
from options import options


class QuantizedResNet(nn.Module):
    def __init__(self, model_fp32):
        super(QuantizedResNet, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.model_fp32 = model_fp32

    def forward(self, x):
        x = self.quant(x)
        x = self.model_fp32(x)
        x = self.dequant(x)
        return x


def train(model, epoch, train_dataloader, test_dataloader, device, args):
    """
    Input:
    model
        pytorch model object to train
    epoch
        int number of total training epochs
    train_dataset
        dataset object which satisfies pytorch dataset format
    test_dataloader
        pytorch dataloader
    device
        string of either 'cuda' or 'cpu'
    """

    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

    best_val_acc = 0
    patience = 0

    for e in range(epoch):
        model.train()

        train_loss = 0
        batch_size = args.batch_size

        with tqdm.tqdm(total=len(train_dataloader) * batch_size, unit='it', unit_scale=True, unit_divisor=1000) as pbar:
            sum_acc = 0
            total = 0
            for X, y in train_dataloader:
                X, y = X.to(device), y.to(device)
                opt.zero_grad()
                logits = model(X)
                loss = criterion(logits, y)

                acc = torch.sum(torch.argmax(logits, dim=1) == y).item()
                loss.backward()
                opt.step()

                # progress bar counter
                train_loss += loss.item()
                pbar.update(batch_size)
                sum_acc += acc
                total += X.shape[0]
                pbar.set_postfix(loss=loss.item(),
                                 acc=sum_acc / total)

        validation_accuracy = eval(model, test_dataloader, device)
        print("Epoch {} - Training loss: {:.4f}, Val Accuracy: {:.4f}".format(e, train_loss/len(train_dataloader),
                                                                              validation_accuracy))

        # Early stopping
        if args.early_stop:
            if validation_accuracy > best_val_acc:
                best_val_acc = validation_accuracy
                patience = 0
            else:
                patience += 1
                if patience > PATIENCE_EPOCH:
                    print("Training early stops at epoch {}".format(e))
                    break


def main():
    args = options()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])

    print(args.dataset)
    print(args.model)

    train_dataset = CustomDataset(args.dataset, './data', train=True, transform=train_transform)
    test_dataset = CustomDataset(args.dataset, './data', train=False, transform=test_transform)

    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)

    model = torch.hub.load('pytorch/vision:v0.10.0', args.model, pretrained=False, num_classes=10).to(device)
    # model = resnet18(num_classes=10, pretrained=False)
    model.eval()
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    quant_model = torch.quantization.fuse_modules(model, [["conv1", "bn1", "relu"]], inplace=True)
    for module_name, module in quant_model.named_children():
        if "layer" in module_name:
            for basic_block_name, basic_block in module.named_children():
                torch.quantization.fuse_modules(
                    basic_block, [["conv1", "bn1", "relu"], ["conv2", "bn2"]],
                    inplace=True)
                for sub_block_name, sub_block in basic_block.named_children():
                    if sub_block_name == "downsample":
                        torch.quantization.fuse_modules(sub_block,
                                                        [["0", "1"]],
                                                        inplace=True)
    quant_model = QuantizedResNet(quant_model)

    quant_model_prepared = torch.quantization.prepare_qat(quant_model.train())
    train(quant_model_prepared, args.epochs, train_loader, test_loader, device, args)


if __name__ == '__main__':
    main()