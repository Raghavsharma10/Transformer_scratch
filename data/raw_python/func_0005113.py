def drawrectangle(self, xa, xb, ya, yb, colour=None, label = None):
        """
        Draws a 1-pixel wide frame AROUND the region you specify. Same convention as for crop().
        
        """
    
        self.checkforpilimage()
        colour = self.defaultcolour(colour)
        self.changecolourmode(colour)
        self.makedraw()
        
        (pilxa, pilya) = self.pilcoords((xa,ya))
        (pilxb, pilyb) = self.pilcoords((xb,yb))
        
        self.draw.rectangle([(pilxa, pilyb-1), (pilxb+1, pilya)], outline = colour)
        
        if label != None:
            # The we write it :
            self.loadlabelfont()
            textwidth = self.draw.textsize(label, font = self.labelfont)[0]
            self.draw.text(((pilxa + pilxb)/2.0 - float(textwidth)/2.0 + 1, pilya + 2), label, fill = colour, font = self.labelfont)