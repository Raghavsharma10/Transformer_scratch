def getScreenDims(self):
        """returns a tuple that contains (screen_width,screen_height)
        """
        width = ale_lib.getScreenWidth(self.obj)
        height = ale_lib.getScreenHeight(self.obj)
        return (width,height)