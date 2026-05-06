def evaluate(expression, start=None, end=None, loader=None, logger=None,
             backend=None, **kwargs):
    '''Evaluate a timeseries ``expression`` into
    an instance of :class:`dynts.dsl.dslresult` which can be used
    to obtain timeseries and/or scatters.
    This is probably the most used function of the library.

    :parameter expression: A timeseries expression string or an instance
        of :class:`dynts.dsl.Expr` obtained using the :func:`~.parse`
        function.
    :parameter start: Start date or ``None``.
    :parameter end: End date or ``None``. If not provided today values is used.
    :parameter loader: Optional :class:`dynts.data.TimeSerieLoader`
        class or instance to use.

        Default ``None``.
    :parameter logger: Optional python logging instance, used if you required
        logging.

        Default ``None``.
    :parameter backend: :class:`dynts.TimeSeries` backend name or ``None``.

    The ``expression`` is parsed and the :class:`~.Symbol` are sent to the
    :class:`dynts.data.TimeSerieLoader` instance for retrieving
    actual timeseries data.
    It returns an instance of :class:`~.DSLResult`.

    Typical usage::

        >>> from dynts import api
        >>> r = api.evaluate('min(GS,window=30)')
        >>> r
        min(GS,window=30)
        >>> ts = r.ts()
    '''
    if isinstance(expression, str):
        expression = parse(expression)
    if not expression or expression.malformed():
        raise CouldNotParse(expression)
    symbols = expression.symbols()
    start = start if not start else todate(start)
    end = end if not end else todate(end)
    data = providers.load(symbols, start, end, loader=loader,
                          logger=logger, backend=backend, **kwargs)
    return DSLResult(expression, data, backend=backend)