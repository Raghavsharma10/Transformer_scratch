def drawcircle(self, x, y, r = 10, colour = None, label = None):
        """
        Draws a circle centered on (x, y) with radius r. All these are in the coordinates of your initial image !
        You give these x and y in the usual ds9 pixels, (0,0) is bottom left.
        I will convert this into the right PIL coordiates.
        """
        
        self.checkforpilimage()
        colour = self.defaultcolour(colour)
        self.changecolourmode(colour)
        self.makedraw()
        
        (pilx, pily) = self.pilcoords((x,y))
        pilr = self.pilscale(r)
        
        self.draw.ellipse([(pilx-pilr+1, pily-pilr+1), (pilx+pilr+1, pily+pilr+1)], outline = colour)
        
        if label != None:
            # The we write it :
            self.loadlabelfont()
            textwidth = self.draw.textsize(label, font = self.labelfont)[0]
            self.draw.text((pilx - float(textwidth)/2.0 + 2, pily + pilr + 4), label, fill = colour, font = self.labelfont)