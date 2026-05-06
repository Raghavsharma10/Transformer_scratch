def load_module(self, name):
        """Load a module from a file.
        """
        # Implementation inspired from pytest.rewrite and importlib

        # If there is an existing module object named 'name' in
        # sys.modules, the loader must use that existing module. (Otherwise,
        # the reload() builtin will not work correctly.)
        if name in sys.modules:
            return sys.modules[name]
        try:
            # we have already done the search, an gone through package layers
            # so we directly feed the latest module and correct path
            # to reuse the logic for choosing the proper loading behavior

            # TODO : double check maybe we do not need the loop here, already handled by finders in dir hierarchy
            # TODO : use exec_module (recent, more tested API) from here
            for name_idx, name_part in enumerate(name.split('.')):
                pkgname = ".".join(name.split('.')[:name_idx+1])
                if pkgname not in sys.modules:
                    if '.' in pkgname:
                        # parent has to be in sys.modules. make sure it is a package, else fails
                        if '__path__' in vars(sys.modules[pkgname.rpartition('.')[0]]):
                            path = sys.modules[pkgname.rpartition('.')[0]].__path__
                        else:
                            raise ImportError("{0} is not a package (no __path__ detected)".format(pkgname.rpartition('.')[0]))
                    else:  # using __file__ instead. should always be there.
                        path = os.path.dirname(sys.modules[pkgname].__file__)if pkgname in sys.modules else None
                    try:
                        file, pathname, description = imp.find_module(pkgname.rpartition('.')[-1], path)
                        sys.modules[pkgname] = imp.load_module(pkgname, file, pathname, description)
                    finally:
                        if file:
                            file.close()
        except:
            # dont pollute the interpreter environment if we dont know what we are doing
            if name in sys.modules:
                del sys.modules[name]
            raise
        return sys.modules[name]