def merge(self, indict):
        """
        A recursive update - useful for merging config files.

        >>> a = '''[section1]
        ...     option1 = True
        ...     [[subsection]]
        ...     more_options = False
        ...     # end of file'''.splitlines()
        >>> b = '''# File is user.ini
        ...     [section1]
        ...     option1 = False
        ...     # end of file'''.splitlines()
        >>> c1 = ConfigObj(b)
        >>> c2 = ConfigObj(a)
        >>> c2.merge(c1)
        >>> c2
        ConfigObj({'section1': {'option1': 'False', 'subsection': {'more_options': 'False'}}})
        """
        for key, val in list(indict.items()):
            if (key in self and isinstance(self[key], dict) and
                                isinstance(val, dict)):
                self[key].merge(val)
            else:
                self[key] = val