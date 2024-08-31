class Airspace(object):
    """
    Create an airspace environment with a given size.
    """
    def __init__(self, size=1000):
        self.area = size ** 2
        self.width = size
        self.length  = size
        self.size = (size, size)
