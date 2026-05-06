def _is_base_matcher_class_definition(meta, classname, dict_):
        """Checks whether given class name and dictionary
        define the :class:`BaseMatcher`.
        """
        if classname != 'BaseMatcher':
            return False
        methods = list(filter(inspect.isfunction, dict_.values()))
        return methods and all(m.__module__ == __name__ for m in methods)