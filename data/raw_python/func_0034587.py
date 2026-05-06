def sadd(self, key, *members):
        """Add the specified members to the set stored at key. Specified
        members that are already a member of this set are ignored. If key does
        not exist, a new set is created before adding the specified members.

        An error is returned when the value stored at key is not a set.

        Returns :data:`True` if all requested members are added. If more
        than one member is passed in and not all members are added, the
        number of added members is returned.

        .. note::

           **Time complexity**: ``O(N)`` where ``N`` is the number of members
           to be added.

        :param key: The key of the set
        :type key: :class:`str`, :class:`bytes`
        :param members: One or more positional arguments to add to the set
        :type key: :class:`str`, :class:`bytes`
        :returns: Number of items added to the set
        :rtype: bool, int

        """
        return self._execute([b'SADD', key] + list(members), len(members))