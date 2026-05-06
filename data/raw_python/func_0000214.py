def _collect_classes(
        self, package_paths: Sequence[str], recurse_subpackages: bool = True
    ) -> Sequence[type]:
        """
        Collect all classes defined in/under ``package_paths``.
        """
        import uqbar.apis

        classes = []
        initial_source_paths: Set[str] = set()
        # Graph source paths and classes
        for path in package_paths:
            try:
                module = importlib.import_module(path)
                if hasattr(module, "__path__"):
                    initial_source_paths.update(getattr(module, "__path__"))
                else:
                    initial_source_paths.add(module.__file__)
            except ModuleNotFoundError:
                path, _, class_name = path.rpartition(".")
                module = importlib.import_module(path)
                classes.append(getattr(module, class_name))
        # Iterate source paths
        for source_path in uqbar.apis.collect_source_paths(
            initial_source_paths, recurse_subpackages=recurse_subpackages
        ):
            package_path = uqbar.apis.source_path_to_package_path(source_path)
            module = importlib.import_module(package_path)
            # Grab any defined classes
            for name in dir(module):
                if name.startswith("_"):
                    continue
                object_ = getattr(module, name)
                if isinstance(object_, type) and object_.__module__ == module.__name__:
                    classes.append(object_)
        return sorted(classes, key=lambda x: (x.__module__, x.__name__))