def find_spec(self, fullname, path, target=None):
        '''finds the appropriate properties (spec) of a module, and sets
           its loader.'''
        if not path:
            path = [os.getcwd()]
        if "." in fullname:
            name = fullname.split(".")[-1]
        else:
            name = fullname
        for entry in path:
            if os.path.isdir(os.path.join(entry, name)):
                # this module has child modules
                filename = os.path.join(entry, name, "__init__.py")
                submodule_locations = [os.path.join(entry, name)]
            else:
                filename = os.path.join(entry, name + ".py")
                submodule_locations = None
            if not os.path.exists(filename):
                continue

            return spec_from_file_location(fullname, filename,
                                           loader=MyLoader(filename),
                                           submodule_search_locations=submodule_locations)
        return None