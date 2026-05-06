def extend_volume(self, volume, size):
        """Extend a volume to a new, larger size.

        :param volume: Name of the volume to be extended.
        :type volume: str
        :type size: int or str
        :param size: Size in bytes, or string representing the size of the
                     volume to be created.

        :returns: A dictionary mapping "name" to volume and "size" to the volume's
                  new size in bytes.
        :rtype: ResponseDict

        .. note::

            The new size must be larger than the volume's old size.

        .. note::

            The maximum volume size supported is 4 petabytes (4 * 2^50).

        .. note::

            If size is an int, it must be a multiple of 512.

        .. note::

           If size is a string, it  must consist of an integer followed by a
           valid suffix.

        Accepted Suffixes

        ====== ======== ======
        Suffix Size     Bytes
        ====== ======== ======
        S      Sector   (2^9)
        K      Kilobyte (2^10)
        M      Megabyte (2^20)
        G      Gigabyte (2^30)
        T      Terabyte (2^40)
        P      Petabyte (2^50)
        ====== ======== ======

        """
        return self.set_volume(volume, size=size, truncate=False)