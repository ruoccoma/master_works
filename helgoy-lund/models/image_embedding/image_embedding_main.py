import os, sys
ROOT_DIR = os.path.dirname((os.path.abspath(os.path.join(os.path.join(__file__, os.pardir), os.pardir)))) + "/"
sys.path.append(ROOT_DIR)

import settings
from inception.cnn_imagenet import run_inception
from vgg.vgg_19 import run_vgg

if settings.IMAGE_EMBEDDING_METHOD == "inception":
	run_inception()
elif settings.IMAGE_EMBEDDING_METHOD == "vgg":
	run_vgg()
