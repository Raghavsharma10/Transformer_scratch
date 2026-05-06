def load(cls, filename, dir=None, main=False, **kwargs):
        """Import a notebook as a module from a filename.
        
        dir: The directory to load the file from.
        main: Load the module in the __main__ context.
        
        > assert Notebook.load('loader.ipynb')
        """
        name = main and "__main__" or Path(filename).stem
        loader = cls(name, str(filename), **kwargs)
        module = module_from_spec(FileModuleSpec(name, loader, origin=loader.path))
        cwd = str(Path(loader.path).parent)
        try:
            with ExitStack() as stack:
                sys.path.append(cwd)
                loader.name != "__main__" and stack.enter_context(_installed_safely(module))
                loader.exec_module(module)
        finally:
            sys.path.pop()

        return module