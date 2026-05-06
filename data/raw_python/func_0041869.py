def setup_subparser(
            self, func=None, setup_as=None, insert_at=None, interprete=True,
            epilog_sections=None, overwrite=False, append_epilog=True,
            return_parser=False, name=None, **kwargs):
        """
        Create a subparser with the name of the given function

        Parameters are the same as for the :meth:`setup_args` function, other
        parameters are parsed to the :meth:`add_subparsers` method if (and only
        if) this method has not already been called.

        Parameters
        ----------
        %(FuncArgParser.setup_args.parameters)s
        return_parser: bool
            If True, the create parser is returned instead of the function
        name: str
            The name of the created parser. If None, the function name is used
            and underscores (``'_'``) are replaced by minus (``'-'``)
        ``**kwargs``
            Any other parameter that is passed to the add_parser method that
            creates the parser

        Other Parameters
        ----------------

        Returns
        -------
        FuncArgParser or %(FuncArgParser.setup_args.returns)s
            If return_parser is True, the created subparser is returned

        Examples
        --------
        Use this method as a decorator::

            >>> from funcargparser import FuncArgParser

            >>> parser = FuncArgParser()

            >>> @parser.setup_subparser
            ... def my_func(my_argument=None):
            ...     pass

            >>> args = parser.parse_args('my-func -my-argument 1'.split())
        """
        def setup(func):
            if self._subparsers_action is None:
                raise RuntimeError(
                    "No subparsers have yet been created! Run the "
                    "add_subparsers method first!")
            # replace underscore by '-'
            name2use = name
            if name2use is None:
                name2use = func.__name__.replace('_', '-')
            kwargs.setdefault('help', docstrings.get_summary(
                docstrings.dedents(inspect.getdoc(func))))
            parser = self._subparsers_action.add_parser(name2use, **kwargs)
            parser.setup_args(
                func, setup_as=setup_as, insert_at=insert_at,
                interprete=interprete, epilog_sections=epilog_sections,
                overwrite=overwrite, append_epilog=append_epilog)
            return func, parser
        if func is None:
            return lambda f: setup(f)[0]
        else:
            return setup(func)[int(return_parser)]