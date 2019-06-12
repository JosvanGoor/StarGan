import os

from argparse import ArgumentParser

def check_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

'''
    if run from this file, parse the configuration
'''
if __name__ == "__main__":
    parser = ArgumentParser(description = "Keras implementation of StarGan")

    # General arguments
    parser.add_argument("--image_size", type = int, default = 128, help = "Image size")
    parser.add_argument("--selected_labels", type = str, nargs = "+", help = "Selected attributes for training",
                        default = ["Blond_Hair", "Brown_Hair", "Black_Hair", "Male", "Young"])
    
    # Training / network parameters
    parser.add_argument("--adv_weight", type = float, default = 1.0, help = "Adverserial weight")
    parser.add_argument("--batch_size", type = int, default = 16, help = "Training batch size")
    parser.add_argument("--cls_weight", type = float, default = 10, help = "Classification weight")
    parser.add_argument("--beta_1", type = float, default = 0.9, help = "Beta1 for Adam optimizer")
    parser.add_argument("--beta_2", type = float, default = 0.999, help = "Beta2 for Adam optimizer")
    parser.add_argument("--dis_hidden", type = int, default = 5, help = "Number of hidden layers in discriminator")
    parser.add_argument("--dis_lr", type = float, default = 0.0001, help = "Discriminator learning rate")
    parser.add_argument("--gen_lr", type = float, default = 0.0001, help = "Generator learning rate")
    parser.add_argument("--gen_residuals", type = int, default = 6, help = "Number of residual layers for generator")
    parser.add_argument("--grad_pen", type = float, default = 10.0, help = "Discriminator's gradient penalty")
    parser.add_argument("--leakyness", type = float, default = 0.01, help = "LeakyReLU leak rate")
    parser.add_argument("--rec_weight", type = float, default = 10.0, help = "Reconstruction weight")

    # Folder parameters
    parser.add_argument("--checkpoint_dir", type = str, default = "checkpoints", help = "Checkpoint save folder")
    parser.add_argument("--data_dir", type = str, default = "data", help = "Location of CelebA folder")
    parser.add_argument("--log_dir", type = str, default = "logs", help = "Logs folder (for tensorboard)")
    parser.add_argument("--model_dir", type = str, default = "models", help = "Completed model folder")

    arguments = parser.parse_args()

    # verify required folders exist
    check_directory(arguments.checkpoint_dir)
    check_directory(arguments.log_dir)
    check_directory(arguments.model_dir)

    # build network