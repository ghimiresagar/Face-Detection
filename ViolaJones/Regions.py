class RectangleRegion:
    """
        The Rectangle region behaves as a part of features.
    """

    def __init__(self, x, y, width, height):
        """
            Initialize the rectangle region.

        :param x: x coordinate of the upper left corner of the rectangle
        :param y: y coordinate of the upper left corner of the rectangle
        :param width: width of the rectangle
        :param height: height of the rectangle
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def computeScore(self, ii):
        return ii[self.y + self.height][self.x + self.width] + ii[self.y][self.x] - (
                    ii[self.y + self.height][self.x] + ii[self.y][self.x + self.width])


    def __str__(self):
        return "(x= %d, y= %d, width= %d, height= %d)" % (self.x, self.y, self.width, self.height)

    def __repr__(self):
        return "RectangleRegion(%d, %d, %d, %d)" % (self.x, self.y, self.width, self.height)