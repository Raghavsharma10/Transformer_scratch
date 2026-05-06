def connect(self):
        """Connect the link sequences of the actual model."""
        try:
            for group in ('inlets', 'receivers', 'outlets', 'senders'):
                self._connect_subgroup(group)
        except BaseException:
            objecttools.augment_excmessage(
                'While trying to build the node connection of the `%s` '
                'sequences of the model handled by element `%s`'
                % (group[:-1], objecttools.devicename(self)))