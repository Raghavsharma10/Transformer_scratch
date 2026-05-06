def remove_device(self, device: Union[DeviceType, str]) -> None:
        """Remove the given |Node| or |Element| object from the actual
        |Nodes| or |Elements| object.

        You can pass either a string or a device:

        >>> from hydpy import Node, Nodes
        >>> nodes = Nodes('node_x', 'node_y')
        >>> node_x, node_y = nodes
        >>> nodes.remove_device(Node('node_y'))
        >>> nodes
        Nodes("node_x")
        >>> nodes.remove_device(Node('node_x'))
        >>> nodes
        Nodes()
        >>> nodes.remove_device(Node('node_z'))
        Traceback (most recent call last):
        ...
        ValueError: While trying to remove the device `node_z` from a \
Nodes object, the following error occurred: The actual Nodes object does \
not handle such a device.

        Method |Devices.remove_device| is disabled for immutable |Nodes|
        and |Elements| objects:

        >>> nodes.mutable = False
        >>> nodes.remove_device('node_z')
        Traceback (most recent call last):
        ...
        RuntimeError: While trying to remove the device `node_z` from a \
Nodes object, the following error occurred: Removing devices from \
immutable Nodes objects is not allowed.
        """
        try:
            if self.mutable:
                _device = self.get_contentclass()(device)
                try:
                    del self._name2device[_device.name]
                except KeyError:
                    raise ValueError(
                        f'The actual {objecttools.classname(self)} '
                        f'object does not handle such a device.')
                del _id2devices[_device][id(self)]
            else:
                raise RuntimeError(
                    f'Removing devices from immutable '
                    f'{objecttools.classname(self)} objects is not allowed.')
        except BaseException:
            objecttools.augment_excmessage(
                f'While trying to remove the device `{device}` from a '
                f'{objecttools.classname(self)} object')