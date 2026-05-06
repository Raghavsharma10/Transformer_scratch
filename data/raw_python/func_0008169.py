def insert(self, **kwargs):
        """
        Insert commands at the beginning of the sequence.

        This is provided because certain commands
        have to come first (such as user creation), but may be need to beadded
        after other commands have already been specified.
        Later calls to insert put their commands before those in the earlier calls.

        Also, since the order of iterated kwargs is not guaranteed (in Python 2.x),
        you should really only call insert with one keyword at a time.  See the doc of append
        for more details.
        :param kwargs: the key/value pair to append first
        :return: the action, so you can append Action(...).insert(...).append(...)
        """
        for k, v in six.iteritems(kwargs):
            self.commands.insert(0, {k: v})
        return self