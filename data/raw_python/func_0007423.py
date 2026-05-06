def getScreenRGB(self,screen_data=None):
        """This function fills screen_data with the data
        screen_data MUST be a numpy array of uint32/int32. This can be initialized like so:
        screen_data = np.array(w*h,dtype=np.uint32)
        Notice, it must be width*height in size also
        If it is None, then this function will initialize it
        """
        if(screen_data is None):
            width = ale_lib.getScreenWidth(self.obj)
            height = ale_lib.getScreenWidth(self.obj)
            screen_data = np.zeros(width*height,dtype=np.uint32)
        ale_lib.getScreenRGB(self.obj,as_ctypes(screen_data))
        return screen_data