def get_double(self, group: str) -> pointerutils.Double:
        """Return the |Double| object appropriate for the given |Element|
        input or output group and the actual |Node.deploymode|.

        Method |Node.get_double| should be of interest for framework
        developers only (and eventually for model developers).

        Let |Node| object `node1` handle different simulation and
        observation values:

        >>> from hydpy import Node
        >>> node = Node('node1')
        >>> node.sequences.sim = 1.0
        >>> node.sequences.obs = 2.0

        The following `test` function shows for a given |Node.deploymode|
        if method |Node.get_double| either returns the |Double| object
        handling the simulated value (1.0) or the |Double| object handling
        the observed value (2.0):

        >>> def test(deploymode):
        ...     node.deploymode = deploymode
        ...     for group in ('inlets', 'receivers', 'outlets', 'senders'):
        ...         print(group, node.get_double(group))

        In the default mode, nodes (passively) route simulated values
        through offering the |Double| object of sequence |Sim| to all
        |Element| input and output groups:

        >>> test('newsim')
        inlets 1.0
        receivers 1.0
        outlets 1.0
        senders 1.0

        Setting |Node.deploymode| to `obs` means that a node receives
        simulated values (from group `outlets` or `senders`), but provides
        observed values (to group `inlets` or `receivers`):

        >>> test('obs')
        inlets 2.0
        receivers 2.0
        outlets 1.0
        senders 1.0

        With |Node.deploymode| set to `oldsim`, the node provides
        (previously) simulated values (to group `inlets` or `receivers`)
        but does not receive any values.  Method |Node.get_double| just
        returns a dummy |Double| object with value 0.0 in this case
        (for group `outlets` or `senders`):

        >>> test('oldsim')
        inlets 1.0
        receivers 1.0
        outlets 0.0
        senders 0.0

        Other |Element| input or output groups are not supported:

        >>> node.get_double('test')
        Traceback (most recent call last):
        ...
        ValueError: Function `get_double` of class `Node` does not support \
the given group name `test`.
        """
        if group in ('inlets', 'receivers'):
            if self.deploymode != 'obs':
                return self.sequences.fastaccess.sim
            return self.sequences.fastaccess.obs
        if group in ('outlets', 'senders'):
            if self.deploymode != 'oldsim':
                return self.sequences.fastaccess.sim
            return self.__blackhole
        raise ValueError(
            f'Function `get_double` of class `Node` does not '
            f'support the given group name `{group}`.')