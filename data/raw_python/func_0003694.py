def lookupmodule(self, filename):
        """Helper function for break/clear parsing -- may be overridden.

        lookupmodule() translates (possibly incomplete) file or module name
        into an absolute file name.
        """
        if os.path.isabs(filename) and os.path.exists(filename):
            return filename
        f = os.path.join(sys.path[0], filename)
        if os.path.exists(f) and self.canonic(f) == self.mainpyfile:
            return f
        root, ext = os.path.splitext(filename)
        origFileName = filename
        if ext == '':
            filename = filename + '.py'
        if os.path.isabs(filename):
            return filename
        for dirname in sys.path:
            while os.path.islink(dirname):
                dirname = os.path.realpath(os.path.join(
                    os.path.dirname(dirname),
                    os.readlink(dirname)))
            fullname = os.path.join(dirname, filename)
            if os.path.exists(fullname):
                return fullname
        if origFileName in sys.modules:
            return sys.modules[origFileName].__file__
        return None