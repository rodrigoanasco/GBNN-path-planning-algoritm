#For mapping
class Point:
    def __init__(self, x, y, z, activity=0, point_type=100.0):
        self.x = x
        self.y = y
        self.z = z
        self.activity = activity
        self.type = point_type  # E, NEG_E, CLEANED
        self.neighbors = []
