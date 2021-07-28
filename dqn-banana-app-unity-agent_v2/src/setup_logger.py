# setup_logger.py
import logging
# import watchtower

logger = logging.getLogger('image2recipe')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('i2r_training.log')
# logger.addHandler(watchtower.CloudWatchLogHandler())
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

logger_goodImages = logging.getLogger('image2recipe_goodimages')
logger_goodImages.setLevel(logging.DEBUG)
fh_gi = logging.FileHandler('i2r_training_goodImages.log')
fh_gi.setLevel(logging.DEBUG)
fh_gi.setFormatter(formatter)
logger_goodImages.addHandler(fh_gi)

logger_badImages = logging.getLogger('image2recipe_badimages')
logger_badImages.setLevel(logging.DEBUG)
fh_bi = logging.FileHandler('i2r_training_badImages.log')
fh_bi.setLevel(logging.DEBUG)
fh_bi.setFormatter(formatter)
logger_badImages.addHandler(fh_bi)

logger_mkdataset = logging.getLogger('image2recipe_mkdataset')
logger_mkdataset.setLevel(logging.DEBUG)
fh_dt = logging.FileHandler('i2r_training_mkdataset.log')
fh_dt.setLevel(logging.DEBUG)
fh_dt.setFormatter(formatter)
logger_mkdataset.addHandler(fh_dt)
