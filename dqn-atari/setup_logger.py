# setup_logger.py
import logging
# import watchtower

logger = logging.getLogger('DQN Visual Agent - v02')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('dqn_visual_training.log')
# logger.addHandler(watchtower.CloudWatchLogHandler())
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)
