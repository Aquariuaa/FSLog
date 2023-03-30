# -*- coding: utf-8 -*-
import sys
sys.dont_write_bytecode = True
import os
import torch
from core.config import Config
from core import Test

PATH = "/.../.../...-bgl_time-tcniniNet-2-5-Feb-27-2023-18-07-28"
VAR_DICT = {
    "test_epoch": 5,
    "device_ids": "0",
    "inner_train_iter": 100,
    "n_gpu": 1,
    "test_episode": 100,
    "batch_size": 64,
    "episode_size": 1,
    # "query_num": 20,
    "test_shot": 5,
    # "shot_num": 20,
}


def main(rank, config):
    test = Test(rank, config, PATH)
    test.test_loop()

if __name__ == "__main__":
    config = Config(os.path.join(PATH, "config.yaml"), VAR_DICT).get_config_dict()
    if config["n_gpu"] > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = config["device_ids"]
        torch.multiprocessing.spawn(main, nprocs=config["n_gpu"], args=(config,))
    else:
        main(0, config)
