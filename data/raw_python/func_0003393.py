def savetostr(self, sortkey = True):
        """
        Save configurations to a single string
        """
        return ''.join(k + '=' + repr(v) + '\n' for k,v in self.config_items(sortkey))