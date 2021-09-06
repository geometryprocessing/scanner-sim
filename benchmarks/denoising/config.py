class Config:
    def __init__(self, path = None):
        self.model = {}

        if path is not None:
            # TODO: Load
            pass

    def __getitem__(self, key):
        keys = key.split('.')

        if len(keys) == 1:
            item = self.model[keys[0]]
        else:
            item = self
            for k in keys:
                item = item[k]

        return item
