def evaluationtable(nodes, criteria, nodenames=None,
                    critnames=None, skip_nan=False):
    """Return a table containing the results of the given evaluation
    criteria for the given |Node| objects.

    First, we define two nodes with different simulation and observation
    data (see function |prepare_arrays| for some explanations):

    >>> from hydpy import pub, Node, nan
    >>> pub.timegrids = '01.01.2000', '04.01.2000', '1d'
    >>> nodes = Node('test1'), Node('test2')
    >>> for node in nodes:
    ...     node.prepare_simseries()
    ...     node.sequences.sim.series = 1.0, 2.0, 3.0
    ...     node.sequences.obs.ramflag = True
    ...     node.sequences.obs.series = 4.0, 5.0, 6.0
    >>> nodes[0].sequences.sim.series = 1.0, 2.0, 3.0
    >>> nodes[0].sequences.obs.series = 4.0, 5.0, 6.0
    >>> nodes[1].sequences.sim.series = 1.0, 2.0, 3.0
    >>> with pub.options.checkseries(False):
    ...     nodes[1].sequences.obs.series = 3.0, nan, 1.0

    Selecting functions |corr| and |bias_abs| as evaluation criteria,
    function |evaluationtable| returns the following table (which is
    actually a pandas data frame):

    >>> from hydpy import evaluationtable, corr, bias_abs
    >>> evaluationtable(nodes, (corr, bias_abs))
           corr  bias_abs
    test1   1.0      -3.0
    test2   NaN       NaN

    One can pass alternative names for both the node objects and the
    criteria functions.  Also, `nan` values can be skipped:

    >>> evaluationtable(nodes, (corr, bias_abs),
    ...                 nodenames=('first node', 'second node'),
    ...                 critnames=('corrcoef', 'bias'),
    ...                 skip_nan=True)
                 corrcoef  bias
    first node        1.0  -3.0
    second node      -1.0   0.0

    The number of assigned node objects and criteria functions must
    match the number of givern alternative names:

    >>> evaluationtable(nodes, (corr, bias_abs),
    ...                 nodenames=('first node',))
    Traceback (most recent call last):
    ...
    ValueError: While trying to evaluate the simulation results of some \
node objects, the following error occurred: 2 node objects are given \
which does not match with number of given alternative names beeing 1.

    >>> evaluationtable(nodes, (corr, bias_abs),
    ...                 critnames=('corrcoef',))
    Traceback (most recent call last):
    ...
    ValueError: While trying to evaluate the simulation results of some \
node objects, the following error occurred: 2 criteria functions are given \
which does not match with number of given alternative names beeing 1.
    """
    if nodenames:
        if len(nodes) != len(nodenames):
            raise ValueError(
                '%d node objects are given which does not match with '
                'number of given alternative names beeing %s.'
                % (len(nodes), len(nodenames)))
    else:
        nodenames = [node.name for node in nodes]
    if critnames:
        if len(criteria) != len(critnames):
            raise ValueError(
                '%d criteria functions are given which does not match '
                'with number of given alternative names beeing %s.'
                % (len(criteria), len(critnames)))
    else:
        critnames = [crit.__name__ for crit in criteria]
    data = numpy.empty((len(nodes), len(criteria)), dtype=float)
    for idx, node in enumerate(nodes):
        sim, obs = prepare_arrays(None, None, node, skip_nan)
        for jdx, criterion in enumerate(criteria):
            data[idx, jdx] = criterion(sim, obs)
    table = pandas.DataFrame(
        data=data, index=nodenames, columns=critnames)
    return table