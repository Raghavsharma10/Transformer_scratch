def copy(self: DevicesTypeBound) -> DevicesTypeBound:
        """Return a shallow copy of the actual |Nodes| or |Elements| object.

        Method |Devices.copy| returns a semi-flat copy of |Nodes| or
        |Elements| objects, due to their devices being not copyable:

        >>> from hydpy import Nodes
        >>> old = Nodes('x', 'y')
        >>> import copy
        >>> new = copy.copy(old)
        >>> new == old
        True
        >>> new is old
        False
        >>> new.devices is old.devices
        False
        >>> new.x is new.x
        True

        Changing the |Device.name| of a device is recognised both by the
        original and the copied collection objects:

        >>> new.x.name = 'z'
        >>> old.z
        Node("z", variable="Q")
        >>> new.z
        Node("z", variable="Q")

        Deep copying is permitted due to the above reason:

        >>> copy.deepcopy(old)
        Traceback (most recent call last):
        ...
        NotImplementedError: Deep copying of Nodes objects is not supported, \
as it would require to make deep copies of the Node objects themselves, \
which is in conflict with using their names as identifiers.
        """
        new = type(self)()
        vars(new).update(vars(self))
        vars(new)['_name2device'] = copy.copy(self._name2device)
        vars(new)['_shadowed_keywords'].clear()
        for device in self:
            _id2devices[device][id(new)] = new
        return new