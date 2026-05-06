def truncate_volume(self, volume, size):
        """Truncate a volume to a new, smaller size.

        :param volume: Name of the volume to truncate.
        :type volume: str
        :param size: Size in bytes, or string representing the size of the
                     volume to be created.
        :type size: int or str

        :returns: A dictionary mapping "name" to volume and "size" to the
                  volume's new size in bytes.
        :rtype: ResponseDict

        .. warnings also::

            Data may be irretrievably lost in this operation.

        .. note::

            A snapshot of the volume in its previous state is taken and
            immediately destroyed, but it is available for recovery for
            the 24 hours following the truncation.

        """
        return self.set_volume(volume, size=size, truncate=True)