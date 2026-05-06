def load(cls, fpath):
        """Loads a module and returns its object.

        :param str|unicode fpath:
        :rtype: module
        """
        module_name = os.path.splitext(os.path.basename(fpath))[0]

        sys.path.insert(0, os.path.dirname(fpath))
        try:
            module = import_module(module_name)

        finally:
            sys.path = sys.path[1:]

        return module