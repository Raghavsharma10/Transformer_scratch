def defaultcolour(self, colour):
        """
        Auxiliary method to choose a default colour.
        Give me a user provided colour : if it is None, I change it to the default colour, respecting negative.
        Plus, if the image is in RGB mode and you give me 128 for a gray, I translate this to the expected (128, 128, 128) ...
        """
        if colour == None:
            if self.negative == True:
                if self.pilimage.mode == "L" :
                    return 0
                else :
                    return (0, 0, 0)
            else :
                if self.pilimage.mode == "L" :
                    return 255
                else :
                    return (255, 255, 255)
        else :
            if self.pilimage.mode == "RGB" and type(colour) == type(0):
                return (colour, colour, colour)
            else :
                return colour