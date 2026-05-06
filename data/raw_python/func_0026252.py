def _list_magic_methods(meta, class_):
        """Return names of magic methods defined by a class.
        :return: Iterable of magic methods, each w/o the ``__`` prefix/suffix
        """
        return [
            name[2:-2] for name, member in class_.__dict__.items()
            if len(name) > 4 and name.startswith('__') and name.endswith('__')
            and inspect.isfunction(member)
        ]