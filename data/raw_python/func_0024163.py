def flush(cls, *args):
        """
        Removes all keys of this namespace
        Without args, clears all keys starting with cls.PREFIX
        if called with args, clears keys starting with given cls.PREFIX + args

        Args:
            *args: Arbitrary number of arguments.

        Returns:
            List of removed keys.
        """
        return _remove_keys([], [(cls._make_key(args) if args else cls.PREFIX) + '*'])