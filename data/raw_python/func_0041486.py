def find_spec(self, fullname, target=None):
        """Try to find a spec for the specified module.  Returns the
        matching spec, or None if not found."""
        is_namespace = False
        tail_module = fullname.rpartition('.')[2]

        base_path = os.path.join(self.path, tail_module)
        for suffix, loader_class in self._loaders:
            init_filename = '__init__' + suffix
            init_full_path = os.path.join(base_path, init_filename)
            full_path = base_path + suffix
            if os.path.isfile(init_full_path):
                return self._get_spec(loader_class, fullname, init_full_path, [base_path], target)
            if os.path.isfile(full_path):  # maybe we need more checks here (importlib filefinder checks its cache...)
                return self._get_spec(loader_class, fullname, full_path, None, target)
        else:
            # If a namespace package, return the path if we don't
            #  find a module in the next section.
            is_namespace = os.path.isdir(base_path)

        if is_namespace:
            _verbose_message('possible namespace for {}'.format(base_path))
            spec = ModuleSpec(fullname, None)
            spec.submodule_search_locations = [base_path]
            return spec
        return None