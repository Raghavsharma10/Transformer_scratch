def append(self, **kwargs):
        """
        Add commands at the end of the sequence.

        Be careful: because this runs in Python 2.x, the order of the kwargs dict may not match
        the order in which the args were specified.  Thus, if you care about specific ordering,
        you must make multiple calls to append in that order.  Luckily, append returns
        the Action so you can compose easily: Action(...).append(...).append(...).
        See also insert, below.
        :param kwargs: the key/value pairs to add
        :return: the action
        """
        for k, v in six.iteritems(kwargs):
            self.commands.append({k: v})
        return self