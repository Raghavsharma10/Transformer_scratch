def prepare_arrays(sim=None, obs=None, node=None, skip_nan=False):
    """Prepare and return two |numpy| arrays based on the given arguments.

    Note that many functions provided by module |statstools| apply function
    |prepare_arrays| internally (e.g. |nse|).  But you can also apply it
    manually, as shown in the following examples.

    Function |prepare_arrays| can extract time series data from |Node|
    objects.  To set up an example for this, we define a initialization
    time period and prepare a |Node| object:

    >>> from hydpy import pub, Node, round_, nan
    >>> pub.timegrids = '01.01.2000', '07.01.2000', '1d'
    >>> node = Node('test')

    Next, we assign values the `simulation` and the `observation` sequences
    (to do so for the `observation` sequence requires a little trick, as
    its values are normally supposed to be read from a file):

    >>> node.prepare_simseries()
    >>> with pub.options.checkseries(False):
    ...     node.sequences.sim.series = 1.0, nan, nan, nan, 2.0, 3.0
    ...     node.sequences.obs.ramflag = True
    ...     node.sequences.obs.series = 4.0, 5.0, nan, nan, nan, 6.0

    Now we can pass the node object to function |prepare_arrays| and
    get the (unmodified) time series data:

    >>> from hydpy import prepare_arrays
    >>> arrays = prepare_arrays(node=node)
    >>> round_(arrays[0])
    1.0, nan, nan, nan, 2.0, 3.0
    >>> round_(arrays[1])
    4.0, 5.0, nan, nan, nan, 6.0

    Alternatively, we can pass directly any iterables (e.g. |list| and
    |tuple| objects) containing the `simulated` and `observed` data:

    >>> arrays = prepare_arrays(sim=list(node.sequences.sim.series),
    ...                         obs=tuple(node.sequences.obs.series))
    >>> round_(arrays[0])
    1.0, nan, nan, nan, 2.0, 3.0
    >>> round_(arrays[1])
    4.0, 5.0, nan, nan, nan, 6.0

    The optional `skip_nan` flag allows to skip all values, which are
    no numbers.  Note that only those pairs of `simulated` and `observed`
    values are returned which do not contain any `nan`:

    >>> arrays = prepare_arrays(node=node, skip_nan=True)
    >>> round_(arrays[0])
    1.0, 3.0
    >>> round_(arrays[1])
    4.0, 6.0

    The final examples show the error messages returned in case of
    invalid combinations of input arguments:

    >>> prepare_arrays()
    Traceback (most recent call last):
    ...
    ValueError: Neither a `Node` object is passed to argument `node` nor \
are arrays passed to arguments `sim` and `obs`.

    >>> prepare_arrays(sim=node.sequences.sim.series, node=node)
    Traceback (most recent call last):
    ...
    ValueError: Values are passed to both arguments `sim` and `node`, \
which is not allowed.

    >>> prepare_arrays(obs=node.sequences.obs.series, node=node)
    Traceback (most recent call last):
    ...
    ValueError: Values are passed to both arguments `obs` and `node`, \
which is not allowed.

    >>> prepare_arrays(sim=node.sequences.sim.series)
    Traceback (most recent call last):
    ...
    ValueError: A value is passed to argument `sim` but \
no value is passed to argument `obs`.

    >>> prepare_arrays(obs=node.sequences.obs.series)
    Traceback (most recent call last):
    ...
    ValueError: A value is passed to argument `obs` but \
no value is passed to argument `sim`.
    """
    if node:
        if sim is not None:
            raise ValueError(
                'Values are passed to both arguments `sim` and `node`, '
                'which is not allowed.')
        if obs is not None:
            raise ValueError(
                'Values are passed to both arguments `obs` and `node`, '
                'which is not allowed.')
        sim = node.sequences.sim.series
        obs = node.sequences.obs.series
    elif (sim is not None) and (obs is None):
        raise ValueError(
            'A value is passed to argument `sim` '
            'but no value is passed to argument `obs`.')
    elif (obs is not None) and (sim is None):
        raise ValueError(
            'A value is passed to argument `obs` '
            'but no value is passed to argument `sim`.')
    elif (sim is None) and (obs is None):
        raise ValueError(
            'Neither a `Node` object is passed to argument `node` nor '
            'are arrays passed to arguments `sim` and `obs`.')
    sim = numpy.asarray(sim)
    obs = numpy.asarray(obs)
    if skip_nan:
        idxs = ~numpy.isnan(sim) * ~numpy.isnan(obs)
        sim = sim[idxs]
        obs = obs[idxs]
    return sim, obs