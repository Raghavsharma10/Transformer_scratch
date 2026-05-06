def coordinate_system(self, coordsys):
        """Sets instrument coordinate system. Accepts an int or string."""
        if coordsys.upper() == "ENU":
            ncs = 0
        elif coordsys.upper() == "XYZ":
            ncs = 1
        elif coordsys.upper() == "BEAM":
            ncs = 2
        elif coordsys in [0, 1, 2]:
            ncs = coordsys
        else:
            raise ValueError("Invalid coordinate system selection")
        self.pdx.CoordinateSystem = ncs