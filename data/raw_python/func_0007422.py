def getScreen(self,screen_data=None):
        """This function fills screen_data with the RAW Pixel data
        screen_data MUST be a numpy array of uint8/int8. This could be initialized like so:
        screen_data = np.array(w*h,dtype=np.uint8)
        Notice, it must be width*height in size also
        If it is None, then this function will initialize it
        Note: This is the raw pixel values from the atari, before any RGB palette transformation takes place
        """
        if(screen_data is None):
            width = ale_lib.getScreenWidth(self.obj)
            height = ale_lib.getScreenWidth(self.obj)
            screen_data = np.zeros(width*height,dtype=np.uint8)
        ale_lib.getScreen(self.obj,as_ctypes(screen_data))
        return screen_data