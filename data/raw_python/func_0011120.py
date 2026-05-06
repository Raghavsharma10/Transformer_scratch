def setslice_with_length(self, startindex, endindex, fs, length):
        """Shim for easily converting old __setitem__ calls"""
        if len(self) < startindex:
            fs = ' '*(startindex - len(self)) + fs
        if len(self) > endindex:
            fs = fs + ' '*(endindex - startindex - len(fs))
            assert len(fs) == endindex - startindex, (len(fs), startindex, endindex)
        result = self.splice(fs, startindex, endindex)
        assert len(result) <= length
        return result