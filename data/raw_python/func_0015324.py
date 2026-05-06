def _assert_struct_type(self, struct, name, types, path=None, extra_info=None):
        """Asserts that given structure is of any of given types.

        Args:
            struct: structure to check
            name: displayable name of the checked structure (e.g. "run_foo" for section run_foo)
            types: list/tuple of types that are allowed for given struct
            path: list with a source file as a first element and previous names
                  (as in name argument to this method) as other elements
            extra_info: extra information to print if error is found (e.g. hint how to fix this)
        Raises:
            YamlTypeError: if given struct is not of any given type; error message contains
                           source file and a "path" (e.g. args -> somearg -> flags) specifying
                           where the problem is
        """
        wanted_yaml_typenames = set()
        for t in types:
            wanted_yaml_typenames.add(self._get_yaml_typename(t))
        wanted_yaml_typenames = ' or '.join(wanted_yaml_typenames)
        actual_yaml_typename = self._get_yaml_typename(type(struct))
        if not isinstance(struct, types):
            err = []
            if path:
                err.append(self._format_error_path(path + [name]))
            err.append('  Expected {w} value for "{n}", got value of type {a}: "{v}"'.
                       format(w=wanted_yaml_typenames,
                              n=name,
                              a=actual_yaml_typename,
                              v=struct))
            if extra_info:
                err.append('Tip: ' + extra_info)
            raise exceptions.YamlTypeError('\n'.join(err))