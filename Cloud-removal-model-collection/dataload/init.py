from .floder_dataset import TrainDataset


def getdata(config):
    train = TrainDataset(config, split="train")
    validation = TrainDataset(config, split="val")
    return train, validation


def get_test_data(config):
    return TrainDataset(config, split="test")
