def showcutoffs(self, redblue = False):
        """
        We use drawmask to visualize pixels above and below the z cutoffs.
        By default this is done in black (above) and white (below) (and adapts to negative images).
        But if you choose redblue = True, I use red for above z2 and blue for below z1.
        """
        
        highmask = self.numpyarray > self.z2
        lowmask = self.numpyarray < self.z1
        if redblue == False :
            if self.negative :
                self.drawmask(highmask, colour = 255)
                self.drawmask(lowmask, colour = 0)
            else :
                self.drawmask(highmask, colour = 0)
                self.drawmask(lowmask, colour = 255)
        else :
            
            self.drawmask(highmask, colour = (255, 0, 0))
            self.drawmask(lowmask, colour = (0, 0, 255))