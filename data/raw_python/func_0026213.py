def find_actual_caller(self):
        """
        Returns the full-qualified module name, full pathname, line number, and
        function in which `StreamTeeLogger.write()` was called.  For example,
        if this instance is used to replace `sys.stdout`, this will return the
        location of any print statement.
        """

        # Gleaned from code in the logging module itself...
        try:
            f = sys._getframe(1)
            ##f = inspect.currentframe(1)
        except Exception:
            f = None
        # On some versions of IronPython, currentframe() returns None if
         # IronPython isn't run with -X:Frames.
        if f is not None:
            f = f.f_back
        rv = "(unknown module)", "(unknown file)", 0, "(unknown function)"
        while hasattr(f, "f_code"):
            co = f.f_code
            filename = os.path.normcase(co.co_filename)
            mod = inspect.getmodule(f)

            if mod is None:
                modname = '__main__'
            else:
                modname = mod.__name__

            if modname == __name__:
                # Crawl back until the first frame outside of this module
                f = f.f_back
                continue

            rv = (modname, filename, f.f_lineno, co.co_name)
            break
        return rv