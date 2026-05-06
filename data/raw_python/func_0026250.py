def _validate_class_definition(meta, classname, bases, dict_):
        """Ensure the matcher class definition is acceptable.
        :raise RuntimeError: If there is a problem
        """
        # let the BaseMatcher class be created without hassle
        if meta._is_base_matcher_class_definition(classname, dict_):
            return

        # ensure that no important magic methods are being overridden
        for name, member in dict_.items():
            if not (name.startswith('__') and name.endswith('__')):
                continue

            # check if it's not a whitelisted magic method name
            name = name[2:-2]
            if not name:
                continue  # unlikely case of a ``____`` function
            if name not in meta._list_magic_methods(BaseMatcher):
                continue
            if name in meta.USER_OVERRIDABLE_MAGIC_METHODS:
                continue

            # non-function attributes, like __slots__, are harmless
            if not inspect.isfunction(member):
                continue

            # classes in this very module are exempt, since they define
            # the very behavior of matchers we want to protect
            if member.__module__ == __name__:
                continue

            raise RuntimeError(
                "matcher class %s cannot override the __%s__ method" % (
                    classname, name))