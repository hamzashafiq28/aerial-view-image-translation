import torch
from model import Generator
from model_RGB import *
from data import *
from PIL import Image
import os
from torchvision import transforms
import numpy as np
from torch.autograd import Variable
#torch.cuda.set_device(1)


def main(generator, custom_dataset, model_path, data_dir, results_dir, input):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Initialize generator
    generator = generator().to(device)
    if device.type == 'cuda':

        model_state_dict = torch.load(model_path)

        # Remove the 'module.' prefix from the keys if present
        if any(key.startswith('module.') for key in model_state_dict.keys()):
            model_state_dict = {key.replace('module.', ''): value for key, value in model_state_dict.items()}

        # Load the modified state dict into the model
        generator.load_state_dict(model_state_dict)


        #generator.load_state_dict(torch.load(model_path))
    else:
        generator.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    # Load test dataset
    # Load test dataset
    if generator == Generator:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256), antialias=True),
    ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((1024, 1024), antialias=True),
        ])
    test_dataset = custom_dataset(root_dir=data_dir, transform=transform, input_domain=input)
    dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)

    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)

    # Perform inference
    with torch.no_grad():
        for step, (input_image, file) in enumerate(dataloader):

            input_image = Variable(input_image).to(device)
            fake_images = generator(input_image)
            # Convert the tensor to a NumPy array
            img_np = fake_images[0].cpu().numpy()
            img_np = np.transpose(img_np, (1, 2, 0))

            # Convert the NumPy array to a Pillow Image
            img_pil = Image.fromarray((img_np * 255).astype(np.uint8))

            # Save the image
            img_pil.save(os.path.join(results_dir, file[0].split('/')[-1]))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Testing script")
    parser.add_argument("--model_path", type=str, default='pretrained/sar2ir/SAR_IR.pth',
                        help="Path to the trained model")
    parser.add_argument("--data_dir", type=str, default='input/sar2ir',
                        help="Path to the directory containing test data")
    parser.add_argument("--results_dir", type=str, default='results/sar2ir',
                        help="Directory to save the results")
    parser.add_argument("--type", type=str, default="SARIR",
                        help="choose a type of input data example SAREO SARRGB, SARIR,  RGBIR")
    args = parser.parse_args()

    Model = {
        "SAREO": Generator,
        "SARRGB": Generator1,  # Change to your custom generator if necessary
        "SARIR": Generator1,
        "RGBIR": Generator1,

    }

    custom_dataset_dict = {

        "SAREO": SAREOTranslationDataset_test,
        "SARRGB": SAREOTranslationDataset_test1,  # Change to your custom generator if necessary
        "SARIR": SAREOTranslationDataset_test1,
        "RGBIR": SAREOTranslationDataset_test1,
        # Add more custom dataset classes if necessary
    }

    input = {

        "SAREO": "SAR",
        "SARRGB": "SAR",  # Change to your custom generator if necessary
        "SARIR": "SAR",
        "RGBIR": "RGB",
        # Add more custom dataset classes if necessary
    }


    main(Model[args.type], custom_dataset_dict[args.type], args.model_path, args.data_dir, args.results_dir, input[args.type])
