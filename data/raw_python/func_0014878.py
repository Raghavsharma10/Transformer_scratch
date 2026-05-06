def _do(self, cmd, *args, **kwargs):
        """Semi-raw execution of a matlab command.

        Smartly handle calls to matlab, figure out what to do with `args`,
        and when to use function call syntax and not.

        If no `args` are specified, the ``cmd`` not ``result = cmd()`` form is
        used in Matlab -- this also makes literal Matlab commands legal
        (eg. cmd=``get(gca, 'Children')``).

        If ``nout=0`` is specified, the Matlab command is executed as
        procedure, otherwise it is executed as function (default), nout
        specifying how many values should be returned (default 1).

        **Beware that if you use don't specify ``nout=0`` for a `cmd` that
        never returns a value will raise an error** (because assigning a
        variable to a call that doesn't return a value is illegal in matlab).


        ``cast`` specifies which typecast should be applied to the result
        (e.g. `int`), it defaults to none.

        XXX: should we add ``parens`` parameter?
        """
        handle_out = kwargs.get('handle_out', _flush_write_stdout)
        #self._session = self._session or mlabraw.open()
        # HACK
        if self._autosync_dirs:
            mlabraw.eval(self._session,  "cd('%s');" % os.getcwd().replace("'", "''"))
        nout =  kwargs.get('nout', 1)
        #XXX what to do with matlab screen output
        argnames = []
        tempargs = []
        try:
            for count, arg in enumerate(args):
                if isinstance(arg, MlabObjectProxy):
                    argnames.append(arg._name)
                else:
                    nextName = 'arg%d__' % count
                    argnames.append(nextName)
                    tempargs.append(nextName)
                    # have to convert these by hand
    ##                 try:
    ##                     arg = self._as_mlabable_type(arg)
    ##                 except TypeError:
    ##                     raise TypeError("Illegal argument type (%s.:) for %d. argument" %
    ##                                     (type(arg), type(count)))
                    mlabraw.put(self._session,  argnames[-1], arg)

            if args:
                cmd = "%s(%s)%s" % (cmd, ", ".join(argnames),
                                    ('',';')[kwargs.get('show',0)])
            # got three cases for nout:
            # 0 -> None, 1 -> val, >1 -> [val1, val2, ...]
            if nout == 0:
                handle_out(mlabraw.eval(self._session, cmd))
                return
            # deal with matlab-style multiple value return
            resSL = ((["RES%d__" % i for i in range(nout)]))
            handle_out(mlabraw.eval(self._session, '[%s]=%s;' % (", ".join(resSL), cmd)))
            res = self._get_values(resSL)

            if nout == 1: res = res[0]
            else:         res = tuple(res)
            if kwargs.has_key('cast'):
                if nout == 0: raise TypeError("Can't cast: 0 nout")
                return kwargs['cast'](res)
            else:
                return res
        finally:
            if len(tempargs) and self._clear_call_args:
                mlabraw.eval(self._session, "clear('%s');" %
                             "','".join(tempargs))