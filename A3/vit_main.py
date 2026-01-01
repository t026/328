import torchvision
import numpy as np
import matplotlib.pyplot as plt

from vit_submission import *

classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

def inspect_data(
    transform: callable,
    n_imgs: int = 5,
    ):
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    indices = np.random.randint(0, len(dataset), size=(n_imgs, ))
    # Visualize with matplotlib
    for i, idx in enumerate(indices):
        img_tensor, label = dataset[idx]
        img = inverse_transform(img_tensor)
        plt.subplot(1, n_imgs, i + 1)
        plt.imshow(img, cmap="gray")
        plt.title(classes[label])

    plt.show()
    del dataset


def main():
    args = Args()
    try:
        import paramparse
        paramparse.process(args)
    except ImportError:
        print("WARNING: You have not installed paramparse. Please manually edit the arguments.")

    # -----
    # NOTE: Always inspect your data
    inspect_data(transform(
        input_resolution=args.input_resolution, 
        mode="train",
    ))

    # -----
    # TODO: Train your ViT model
    #train_vit_model(args)
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = ViT(args.num_classes, args.input_resolution, args.patch_size, args.in_channels, args.hidden_size, args.layers, args.heads)
    checkpoint = torch.load("best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])    
    test_classification_model(model, test_loader)   


if __name__ == "__main__":
    main()