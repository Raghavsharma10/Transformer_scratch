def set_app_args(self, *args):
        """Sets ``sys.argv`` for python apps.

        Examples:
            * pyargv="one two three" will set ``sys.argv`` to ``('one', 'two', 'three')``.

        :param args:
        """
        if args:
            self._set('pyargv', ' '.join(args))

        return self._section