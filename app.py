from src.imagedata import ImageData
from src.model import Model

imgdat = ImageData \
(
    "data/images/",
    "data/list_attr_celeba.csv",
    (128, 128),
    8,
)

model = Model \
(
    "models/",
    "logs/",
    imgdat
)