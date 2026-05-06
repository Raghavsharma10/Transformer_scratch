def setitem(self, startindex, fs):
        """Shim for easily converting old __setitem__ calls"""
        return self.setslice_with_length(startindex, startindex+1, fs, len(self))