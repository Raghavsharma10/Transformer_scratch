def setup_args(self, func=None, setup_as=None, insert_at=None,
                   interprete=True, epilog_sections=None,
                   overwrite=False, append_epilog=True):
        """
        Add the parameters from the given `func` to the parameter settings

        Parameters
        ----------
        func: function
            The function to use. If None, a function will be returned that can
            be used as a decorator
        setup_as: str
            The attribute that shall be assigned to the function in the
            resulting namespace. If specified, this function will be used when
            calling the :meth:`parse2func` method
        insert_at: int
            The position where the given `func` should be inserted. If None,
            it will be appended at the end and used when calling the
            :meth:`parse2func` method
        interprete: bool
            If True (default), the docstrings are interpreted and switches and
            lists are automatically inserted (see the
            [interpretation-docs]_
        epilog_sections: list of str
            The headers of the sections to extract. If None, the
            :attr:`epilog_sections` attribute is used
        overwrite: bool
            If True, overwrite the existing epilog and the existing description
            of the parser
        append_epilog: bool
            If True, append to the existing epilog

        Returns
        -------
        function
            Either the function that can be used as a decorator (if `func` is
            ``None``), or the given `func` itself.

        Examples
        --------
        Use this method as a decorator::

            >>> @parser.setup_args
            ... def do_something(a=1):
                '''
                Just an example

                Parameters
                ----------
                a: int
                    A number to increment by one
                '''
                return a + 1
            >>> args = parser.parse_args('-a 2'.split())

        Or by specifying the setup_as function::

            >>> @parser.setup_args(setup_as='func')
            ... def do_something(a=1):
                '''
                Just an example

                Parameters
                ----------
                a: int
                    A number to increment by one
                '''
                return a + 1
            >>> args = parser.parse_args('-a 2'.split())
            >>> args.func is do_something
            >>> parser.parse2func('-a 2'.split())
            3

        References
        ----------
        .. [interpretation-docs]
           http://funcargparse.readthedocs.io/en/latest/docstring_interpretation.html)
        """
        def setup(func):
            # insert the function
            if insert_at is None:
                self._used_functions.append(func)
            else:
                self._used_functions.insert(insert_at, func)

            args_dict = self.unfinished_arguments

            # save the function to use in parse2funcs
            if setup_as:
                args_dict[setup_as] = dict(
                    long=setup_as, default=func, help=argparse.SUPPRESS)
                self._setup_as = setup_as

            # create arguments
            args, varargs, varkw, defaults = inspect.getargspec(func)
            full_doc = docstrings.dedents(inspect.getdoc(func))

            summary = docstrings.get_full_description(full_doc)
            if summary:
                if not self.description or overwrite:
                    self.description = summary
                full_doc = docstrings._remove_summary(full_doc)

            self.extract_as_epilog(full_doc, epilog_sections, overwrite,
                                   append_epilog)

            doc = docstrings._get_section(full_doc, 'Parameters') + '\n'
            doc += docstrings._get_section(full_doc, 'Other Parameters')
            doc = doc.rstrip()
            default_min = len(args or []) - len(defaults or [])
            for i, arg in enumerate(args):
                if arg == 'self' or arg in args_dict:
                    continue
                arg_doc, dtype = self.get_param_doc(doc, arg)
                args_dict[arg] = d = {'dest': arg, 'short': arg.replace('_',
                                                                        '-'),
                                      'long': arg.replace('_', '-')}
                if arg_doc:
                    d['help'] = arg_doc
                    if i >= default_min:
                        d['default'] = defaults[i - default_min]
                    else:
                        d['positional'] = True
                    if interprete and dtype == 'bool' and 'default' in d:
                        d['action'] = 'store_false' if d['default'] else \
                            'store_true'
                    elif interprete and dtype:
                        if dtype.startswith('list of'):
                            d['nargs'] = '+'
                            dtype = dtype[7:].strip()
                        if dtype in ['str', 'string', 'strings']:
                            d['type'] = six.text_type
                            if dtype == 'strings':
                                dtype = 'string'
                        else:
                            try:
                                d['type'] = getattr(builtins, dtype)
                            except AttributeError:
                                try:    # maybe the dtype has a final 's'
                                    d['type'] = getattr(builtins, dtype[:-1])
                                    dtype = dtype[:-1]
                                except AttributeError:
                                    pass
                        d['metavar'] = dtype
            return func
        if func is None:
            return setup
        else:
            return setup(func)