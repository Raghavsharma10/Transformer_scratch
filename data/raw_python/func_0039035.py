def _get_fqn(cls, path):
        """get full qualified name as list of strings
        :return: (list - str) of path segments from top package to given path
        """
        name_list = [path.stem]
        current_path = path
        # move to parent path until parent path is a python package
        while cls.is_pkg(current_path.parent):
            name_list.append(current_path.parent.name)
            current_path = current_path.parent
        return list(reversed(name_list))