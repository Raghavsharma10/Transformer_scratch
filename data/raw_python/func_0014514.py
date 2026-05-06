def create_volume(self, volume, size, **kwargs):
        """Create a volume and return a dictionary describing it.

        :param volume: Name of the volume to be created.
        :type volume: str
        :param size: Size in bytes, or string representing the size of the
                     volume to be created.
        :type size: int or str
        :param \*\*kwargs: See the REST API Guide on your array for the
                           documentation on the request:
                           **POST volume/:volume**
        :type \*\*kwargs: optional

        :returns: A dictionary describing the created volume.
        :rtype: ResponseDict

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
        data = {"size": size}
        data.update(kwargs)
        return self._request("POST", "volume/{0}".format(volume), data)