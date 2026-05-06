def getRAM(self,ram=None):
        """This function grabs the atari RAM.
        ram MUST be a numpy array of uint8/int8. This can be initialized like so:
        ram = np.array(ram_size,dtype=uint8)
        Notice: It must be ram_size where ram_size can be retrieved via the getRAMSize function.
        If it is None, then this function will initialize it.
        """
        if(ram is None):
            ram_size = ale_lib.getRAMSize(self.obj)
            ram = np.zeros(ram_size,dtype=np.uint8)
        ale_lib.getRAM(self.obj,as_ctypes(ram))