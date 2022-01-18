# from data import mnist
import argparse
from dotenv import find_dotenv, load_dotenv

import torch
from torch.utils.data import DataLoader, TensorDataset

class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        getattr(self, "evaluate")()
    

    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('load_model_from', default="")
        parser.add_argument('load_images_from', default="")
        parser.add_argument('load_labels_from', default="")  
        # add any additional argument that you want
        args = parser.parse_args() 
        print(args)
        
        # TODO-Done: Implement evaluation logic here
        model = torch.load(args.load_model_from)
        test_imgs = torch.load(args.load_images_from)
        test_labs = torch.load(args.load_labels_from)        
        test_ds = TensorDataset(test_imgs, test_labs)
        test_dl = DataLoader(test_ds, batch_size=10, shuffle=False)

        test_correct = 0  # Number of correct predictions on the test set
        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():
            for images, labels in test_dl:
                log_ps = model(images)
                ps = torch.exp(log_ps)
                _, top_class = ps.topk(1, dim=1)
                print('True label:', labels.numpy().flatten())
                print('Predicted :', top_class.numpy().flatten())
                equals = top_class == labels.view(*top_class.shape)                
                test_correct += equals.sum().item()
        print("Accuracy: {:.3f}".format(test_correct / len(test_dl.dataset)))

if __name__ == '__main__':
    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())


    TrainOREvaluate()

# run
# python src/models/predict_model.py  models/my_trained_model.pt  data/processed/example_images.pt  data/processed/example_labels.pt