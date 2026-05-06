def savetofile(self, filelike, sortkey = True):
        """
        Save configurations to a file-like object which supports `writelines`
        """
        filelike.writelines(k + '=' + repr(v) + '\n' for k,v in self.config_items(sortkey))