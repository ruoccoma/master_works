import settings
from inception.cnn_imagenet import run_inception
from vgg.vgg_19 import run_vgg

if settings.IMAGE_EMBEDDING_METHOD == "inception":
	run_inception()
elif settings.IMAGE_EMBEDDING_METHOD == "vgg":
	run_vgg()
