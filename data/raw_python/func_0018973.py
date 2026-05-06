def add_device(self, device: Union[DeviceType, str]) -> None:
        """Add the given |Node| or |Element| object to the actual
        |Nodes| or |Elements| object.

        You can pass either a string or a device:

        >>> from hydpy import Nodes
        >>> nodes = Nodes()
        >>> nodes.add_device('old_node')
        >>> nodes
        Nodes("old_node")
        >>> nodes.add_device('new_node')
        >>> nodes
        Nodes("new_node", "old_node")

        Method |Devices.add_device| is disabled for immutable |Nodes|
        and |Elements| objects:

        >>> nodes.mutable = False
        >>> nodes.add_device('newest_node')
        Traceback (most recent call last):
        ...
        RuntimeError: While trying to add the device `newest_node` to a \
Nodes object, the following error occurred: Adding devices to immutable \
Nodes objects is not allowed.
        """
        try:
            if self.mutable:
                _device = self.get_contentclass()(device)
                self._name2device[_device.name] = _device
                _id2devices[_device][id(self)] = self
            else:
                raise RuntimeError(
                    f'Adding devices to immutable '
                    f'{objecttools.classname(self)} objects is not allowed.')
        except BaseException:
            objecttools.augment_excmessage(
                f'While trying to add the device `{device}` to a '
                f'{objecttools.classname(self)} object')