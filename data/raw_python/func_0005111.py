def drawpoint(self, x, y, colour = None):
        """
        Most elementary drawing, single pixel, used mainly for testing purposes.
        Coordinates are those of your initial image !
        """
        self.checkforpilimage()
        colour = self.defaultcolour(colour)
        self.changecolourmode(colour)
        self.makedraw()
        
        (pilx, pily) = self.pilcoords((x,y))
        
        self.draw.point((pilx, pily), fill = colour)