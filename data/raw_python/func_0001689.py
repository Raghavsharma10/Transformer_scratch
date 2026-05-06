def get_imported_module_from_file(file_path):
        """ import module from python file path and return imported module
        """
        if p_compat.is_py3:
            imported_module = importlib.machinery.SourceFileLoader('module_name', file_path).load_module()
        elif p_compat.is_py2:
            imported_module = imp.load_source('module_name', file_path)
        else:
            raise RuntimeError("Neither Python 3 nor Python 2.")
    
        return imported_module