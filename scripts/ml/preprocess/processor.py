import pandas as pd
from .move_average import MoveAverage

class Processor:
    def __init__(self, DataHandler, ymd):
        self.DataHandler = DataHandler(ymd)

    def run(self):
        self.DataHandler.merge()
        df = self.DataHandler.process()
        return self.DataHandler.data_split(df)
