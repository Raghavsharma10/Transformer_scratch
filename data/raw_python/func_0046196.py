def _build_dependencies(self):
        """
        >>> CodeBaseDoc(['examples'])['subclass.js'].module.all_dependencies
        ['module.js', 'module_closure.js', 'class.js', 'subclass.js']
        """
        for module in list(self.values()):
            module.set_all_dependencies(find_dependencies([module.name], self))