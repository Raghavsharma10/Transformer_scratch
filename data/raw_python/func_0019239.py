def outdated(self):
        """True if at least one of the |Cythonizer.pysourcefiles| is
        newer than the compiled file under |Cythonizer.pyxfilepath|,
        otherwise False.
        """
        if hydpy.pub.options.forcecompiling:
            return True
        if os.path.split(hydpy.__path__[0])[-2].endswith('-packages'):
            return False
        if not os.path.exists(self.dllfilepath):
            return True
        cydate = os.stat(self.dllfilepath).st_mtime
        for pysourcefile in self.pysourcefiles:
            pydate = os.stat(pysourcefile).st_mtime
            if pydate > cydate:
                return True
        return False