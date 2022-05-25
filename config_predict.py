import json


class Config:
    def __init__(self, args):
        with open(args, "r", encoding="utf-8") as f:
            config = json.load(f)

        self.dataset = config['dataset']
        self.batch_size = config['batch_size']

    def __repr__(self):
        return "{}".format(self.__dict__.items())
