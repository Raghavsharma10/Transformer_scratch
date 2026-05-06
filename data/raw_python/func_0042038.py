def find_tabs(self, custom_table_classes=None):
        """Finds all classes that are subcalss of Table and loads them into
         a dictionary named tables."""
        for module_name in get_all_modules(self.package_path):
            for name, _type in get_all_classes(module_name):
                # pylint: disable=W0640
                subclasses = [Table] + (custom_table_classes or list())
                iss_subclass = map(lambda c: issubclass(_type, c), subclasses)
                if isclass(_type) and any(iss_subclass):
                    self.tabs.update([[name, _type]])