import oyaml as yaml

class Config(object):
    def __init__(self, dct):
        self.dict = dct

    def __iter__(self):
        return iter(self.dict)

    def __repr__(self):
        return repr(self.dict)

    def __str__(self):
        return str(self.dict)

    def __getattr__(self, attr):
        try:
            val = self.dict[attr]
            if isinstance(val, dict):
                val = Config(val)
            return val
        except KeyError:
            raise AttributeError
