import argparse

class BaseParser():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    def parse(self):
        self.parser.add_argument("--test_name", default="run", help="name of test file")
        self.parser.add_argument("--model", default="PSNR", help="name of model")
        return self.parser.parse_args()
