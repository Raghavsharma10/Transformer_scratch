def identifier(self, camelsplit=False, ascii=True):
        """return a python identifier from the string (underscore separators)"""
        return self.nameify(camelsplit=camelsplit, ascii=ascii, sep='_')