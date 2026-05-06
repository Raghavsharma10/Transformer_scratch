def makedraw(self):
        """Auxiliary method to make a draw object if not yet done.
        This is also called by changecolourmode, when we go from L to RGB, to get a new draw object.
        """
        if self.draw == None:
            self.draw = imdw.Draw(self.pilimage)