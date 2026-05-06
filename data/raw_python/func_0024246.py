def _import_object(self, path, look_for_cls_method):
        """
        Imports the module that contains the referenced method.

        Args:
            path: python path of class/function
            look_for_cls_method (bool): If True, treat the last part of path as class method.

        Returns:
            Tuple. (class object, class name, method to be called)

        """
        last_nth = 2 if look_for_cls_method else 1
        path = path.split('.')
        module_path = '.'.join(path[:-last_nth])
        class_name = path[-last_nth]
        module = importlib.import_module(module_path)
        if look_for_cls_method and path[-last_nth:][0] == path[-last_nth]:
            class_method = path[-last_nth:][1]
        else:
            class_method = None
        return getattr(module, class_name), class_name, class_method