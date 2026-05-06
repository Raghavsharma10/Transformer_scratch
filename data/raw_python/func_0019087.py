def remove(self, *values):
        """Remove the defined variables.

        The variables to be removed can be selected in two ways.  But the
        first example shows that passing nothing or an empty iterable to
        method |Variable2Auxfile.remove| does not remove any variable:

        >>> from hydpy import dummies
        >>> v2af = dummies.v2af
        >>> v2af.remove()
        >>> v2af.remove([])
        >>> from hydpy import print_values
        >>> print_values(v2af.filenames)
        file1, file2
        >>> print_values(v2af.variables, width=30)
        eqb(5000.0), eqb(10000.0),
        eqd1(100.0), eqd2(50.0),
        eqi1(2000.0), eqi2(1000.0)

        The first option is to pass auxiliary file names:

        >>> v2af.remove('file1')
        >>> print_values(v2af.filenames)
        file2
        >>> print_values(v2af.variables)
        eqb(10000.0), eqd1(100.0), eqd2(50.0)

        The second option is, to pass variables of the correct type
        and value:

        >>> v2af = dummies.v2af
        >>> v2af.remove(v2af.eqb[0])
        >>> print_values(v2af.filenames)
        file1, file2
        >>> print_values(v2af.variables)
        eqb(10000.0), eqd1(100.0), eqd2(50.0), eqi1(2000.0), eqi2(1000.0)

        One can pass multiple variables or iterables containing variables
        at once:

        >>> v2af = dummies.v2af
        >>> v2af.remove(v2af.eqb, v2af.eqd1, v2af.eqd2)
        >>> print_values(v2af.filenames)
        file1
        >>> print_values(v2af.variables)
        eqi1(2000.0), eqi2(1000.0)

        Passing an argument that equals neither a registered file name or a
        registered variable results in the following exception:

        >>> v2af.remove('test')
        Traceback (most recent call last):
        ...
        ValueError: While trying to remove the given object `test` of type \
`str` from the actual Variable2AuxFile object, the following error occurred: \
`'test'` is neither a registered filename nor a registered variable.
        """
        for value in objecttools.extract(values, (str, variabletools.Variable)):
            try:
                deleted_something = False
                for fn2var in list(self._type2filename2variable.values()):
                    for fn_, var in list(fn2var.items()):
                        if value in (fn_, var):
                            del fn2var[fn_]
                            deleted_something = True
                if not deleted_something:
                    raise ValueError(
                        f'`{repr(value)}` is neither a registered '
                        f'filename nor a registered variable.')
            except BaseException:
                objecttools.augment_excmessage(
                    f'While trying to remove the given object `{value}` '
                    f'of type `{objecttools.classname(value)}` from the '
                    f'actual Variable2AuxFile object')