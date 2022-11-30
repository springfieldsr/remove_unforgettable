from utils import *
from options import options
import time, random

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

    # train_dataset = CustomDataset(args.dataset, './data', train=True, transform=train_transform)
    train_dataset = ShuffledDataset(args.dataset, './data', args.top_k * args.label_shuffle, train=True, download=True, transform=train_transform, noise_type=args.noise_type)
    saved_dest = DumpIndicesToFile(list(train_dataset.get_shuffle_mapping().keys()), args.dataset+"_noisy_indices", expr_path)
    print("Indices of noisy data are saved to " + saved_dest)
    test_dataset = CustomDataset(args.dataset, './data', train=False, transform=test_transform)

    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)

    if args.remove_indices: # 
        remove_indices = [int(x) for x in open(args.remove_indices).read().splitlines()]
        if args.remove_fraction: # remove only 90% of the memorized indices
            remove_indices = set(random.sample(remove_indices, int(0.9*len(remove_indices))))
        else: # remove all the memorized indices
            remove_indices = set(remove_indices)
        train_indices = [i for i in range(len(train_dataset)) if i not in remove_indices]
        train_dataset = Subset(train_dataset, train_indices)
    # model = torchvision.models.resnet18().to(device)
    # TODO: pretrained or not matters or not? Not sure
    model = torch.hub.load('pytorch/vision:v0.10.0', args.model, pretrained=False).to(device)

    start = time.time()
    if not args.return_memorized_fraction:
        remove_indices = train(model, args.epochs, train_dataset, test_loader, device, args, start)
        saved_dest = DumpIndicesToFile(remove_indices, args.dataset, expr_path)
        print("Indices of memorized indices are saved to " + saved_dest)
    else:
        fraction, truly_unforgettable = train(model, args.epochs, train_dataset, test_loader, device, args, start)
        print("{} of the marked data points are actually never forgetten".format(fraction))
        saved_dest = DumpIndicesToFile(truly_unforgettable, args.dataset, expr_path)
        print("Indices of truly unforgettable data are saved to " + saved_dest)
    print(f"Total time elpased : {time.time()-start:.2f} ")
    # cleanse the dataset and retrain
    # model = torch.hub.load('pytorch/vision:v0.10.0', args.model, pretrained=True).to(device)
    # train_dataset.cleanse(pred_indices)
    # train(model, args.epochs, False, train_dataset, test_loader, device, args)


if __name__ == '__main__':
    main()
