def getSaveFileName(*args):
        """
        Normalizes the getSaveFileName method between the different Qt
        wrappers.
        
        :return     (<str> filename, <bool> accepted)
        """
        result = QtGui.QFileDialog.getSaveFileName(*args)
        
        # PyQt4 returns just a string
        if type(result) is not tuple:
            return result, bool(result)
        
        # PySide returns a tuple of str, bool
        else:
            return result