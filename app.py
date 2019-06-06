from src.model import create_generator, create_discriminator
from src.imagedata import ImageData
from src.utility import read_image, generate_label_layers

# create_generator(128, 5, 6).summary()
# create_discriminator(128, 5).summary()

imagedata = ImageData("data/images/", "data/list_attr_celeba.csv", 128, 8)
imagedata.read_attribute_file()

batch = imagedata.get_train_batch(4)
print(batch[0][0])
read_image(imagedata.image_dir, batch[0][0], 128)

layers = generate_label_layers(batch[0][1], 128)

