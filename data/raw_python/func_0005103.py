def irafcrop(self, irafcropstring):
        """
        This is a wrapper around crop(), similar to iraf imcopy,
        using iraf conventions (100:199 will be 100 pixels, not 99).
        """
        irafcropstring = irafcropstring[1:-1] # removing the [ ]
        ranges = irafcropstring.split(",")
        xr = ranges[0].split(":")
        yr = ranges[1].split(":")
        xmin = int(xr[0])
        xmax = int(xr[1])+1
        ymin = int(yr[0])
        ymax = int(yr[1])+1
        self.crop(xmin, xmax, ymin, ymax)