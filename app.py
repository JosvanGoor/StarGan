import argparse
import os
from src.network import Network

"""parsing and configuration"""
if __name__ == "__main__":
    desc = "Tensorflow implementation of StarGAN"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train', help='train or test ?')
    parser.add_argument('--dataset', type=str, default='data', help='dataset_name')

    parser.add_argument('--epoch', type=int, default=20, help='The number of epochs to run')
    parser.add_argument('--iteration', type=int, default=10000, help='The number of training iterations')
    parser.add_argument('--batch_size', type=int, default=16, help='The size of batch size')
    parser.add_argument('--print_freq', type=int, default=1000, help='The number of image_print_freq')
    parser.add_argument('--save_freq', type=int, default=1000, help='The number of ckpt_save_freq')
    parser.add_argument('--decay_flag', type=lambda x: x.lower() == "true", default=True, help='The decay_flag')
    parser.add_argument('--decay_epoch', type=int, default=10, help='decay epoch')

    parser.add_argument('--lr', type=float, default=0.0001, help='The learning rate')
    parser.add_argument('--ld', type=float, default=10.0, help='The gradient penalty lambda')
    parser.add_argument('--adv_weight', type=float, default=1, help='Weight about GAN')
    parser.add_argument('--rec_weight', type=float, default=10, help='Weight about Reconstruction')
    parser.add_argument('--cls_weight', type=float, default=10, help='Weight about Classification')

    parser.add_argument('--gan_type', type=str, default='wgan-gp', help='gan / lsgan / wgan-gp / wgan-lp / dragan / hinge')
    parser.add_argument('--selected_attrs', type=str, nargs='+', help='selected attributes for the CelebA dataset',
                        default=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'])

    parser.add_argument('--custom_label', type=int, nargs='+', help='custom label about selected attributes',
                        default=[1, 0, 0, 0, 0])
    # If your image is "Young, Man, Black Hair" = [1, 0, 0, 1, 1]

    parser.add_argument('--ch', type=int, default=64, help='base channel number per layer')
    parser.add_argument('--n_res', type=int, default=6, help='The number of resblock')
    parser.add_argument('--n_dis', type=int, default=5, help='The number of discriminator layer')
    parser.add_argument('--n_critic', type=int, default=5, help='The number of critic')

    parser.add_argument('--img_size', type=int, default=128, help='The size of image')
    parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')
    parser.add_argument('--augment_flag', type=lambda x: x.lower() == "true", default=True, help='Image augmentation use or not')

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--sample_dir', type=str, default='samples',
                        help='Directory name to save the samples on training')
    parser.add_argument("--model_dir", type=str, default="models",
                        help="Model save directory")

    arguments = parser.parse_args()

    # Create directories if not exist.
    if not os.path.exists(arguments.log_dir):
        os.makedirs(arguments.log_dir)
    if not os.path.exists(arguments.model_dir):
        os.makedirs(arguments.model_dir)
    if not os.path.exists(arguments.sample_dir):
        os.makedirs(arguments.sample_dir)
    if not os.path.exists(arguments.result_dir):
        os.makedirs(arguments.result_dir)

    network = Network(arguments)
    network.build()

    network.train()