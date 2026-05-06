def as_partition(self, partition=0, **kwargs):
        """Return a new PartitionIdentity based on this Identity.

        :param partition: Integer partition number for PartitionObjectNumber
        :param kwargs:

        """

        assert isinstance(self._name, Name), "Wrong type: {}".format(type(self._name))
        assert isinstance(self._on, DatasetNumber), "Wrong type: {}".format(type(self._on))

        name = self._name.as_partition(**kwargs)
        on = self._on.as_partition(partition)

        return PartitionIdentity(name, on)