def getDirectory(*args):
        """
        Normalizes the getDirectory method between the different Qt
        wrappers.
        
        :return     (<str> filename, <bool> accepted)
        """
        result = QtGui.QFileDialog.getDirectory(*args)
        
        # PyQt4 returns just a string
        if type(result) is not tuple:
            return result, bool(result)
        
        # PySide returns a tuple of str, bool
        else:
            return result