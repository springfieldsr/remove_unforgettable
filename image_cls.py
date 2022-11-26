from utils import *
from options import options
import time

def main():
    # see options.py
    
    args = options()
    print(args.early_stop)
    expr_path = GenerateEnvironment(args)
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

    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)

    # model = torchvision.models.resnet18().to(device)
    # TODO: pretrained or not matters or not? Not sure
    model = torch.hub.load('pytorch/vision:v0.10.0', args.model, pretrained=False).to(device)

    start = time.time()
    if not args.return_memorized_fraction:
        remove_indices = train(model, args.epochs, train_dataset, test_loader, device, args)
        saved_dest = DumpIndicesToFile(remove_indices, args.dataset, expr_path)
        print("Indices of memorized indices are saved to " + saved_dest)
    else:
        fraction = train(model, args.epochs, train_dataset, test_loader, device, args)
        print("{} of the marked data points are actually never forgetten".format(fraction))
    print(f"Total time elpased : {time.time()-start:.2f} ")
    # cleanse the dataset and retrain
    # model = torch.hub.load('pytorch/vision:v0.10.0', args.model, pretrained=True).to(device)
    # train_dataset.cleanse(pred_indices)
    # train(model, args.epochs, False, train_dataset, test_loader, device, args)


if __name__ == '__main__':
    main()
