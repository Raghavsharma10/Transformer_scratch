def update_masters(self):
        """Update all `master` |Substituter| objects.

        If a |Substituter| object is passed to the constructor of another
        |Substituter| object, they become `master` and `slave`:

        >>> from hydpy.core.autodoctools import Substituter
        >>> sub1 = Substituter()
        >>> from hydpy.core import devicetools
        >>> sub1.add_module(devicetools)
        >>> sub2 = Substituter(sub1)
        >>> sub3 = Substituter(sub2)
        >>> sub3.master.master is sub1
        True
        >>> sub2 in sub1.slaves
        True

        During initialization, all mappings handled by the master object
        are passed to its new slave:

        >>> sub3.find('Node|')
        |Node| :class:`~hydpy.core.devicetools.Node`
        |devicetools.Node| :class:`~hydpy.core.devicetools.Node`

        Updating a slave, does not affect its master directly:

        >>> from hydpy.core import hydpytools
        >>> sub3.add_module(hydpytools)
        >>> sub3.find('HydPy|')
        |HydPy| :class:`~hydpy.core.hydpytools.HydPy`
        |hydpytools.HydPy| :class:`~hydpy.core.hydpytools.HydPy`
        >>> sub2.find('HydPy|')

        Through calling |Substituter.update_masters|, the `medium2long`
        mappings are passed from the slave to its master:

        >>> sub3.update_masters()
        >>> sub2.find('HydPy|')
        |hydpytools.HydPy| :class:`~hydpy.core.hydpytools.HydPy`

        Then each master object updates its own master object also:

        >>> sub1.find('HydPy|')
        |hydpytools.HydPy| :class:`~hydpy.core.hydpytools.HydPy`

        In reverse, subsequent updates of master objects to not affect
        their slaves directly:

        >>> from hydpy.core import masktools
        >>> sub1.add_module(masktools)
        >>> sub1.find('Masks|')
        |Masks| :class:`~hydpy.core.masktools.Masks`
        |masktools.Masks| :class:`~hydpy.core.masktools.Masks`
        >>> sub2.find('Masks|')

        Through calling |Substituter.update_slaves|, the `medium2long`
        mappings are passed the master to all of its slaves:

        >>> sub1.update_slaves()
        >>> sub2.find('Masks|')
        |masktools.Masks| :class:`~hydpy.core.masktools.Masks`
        >>> sub3.find('Masks|')
        |masktools.Masks| :class:`~hydpy.core.masktools.Masks`
        """
        if self.master is not None:
            self.master._medium2long.update(self._medium2long)
            self.master.update_masters()