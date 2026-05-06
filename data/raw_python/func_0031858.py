def _process_arguments(self, arguments):
        """Process the arguments

        :param arguments: arguments string of a message
        :type arguments: :class:`str` | None
        :returns: A list of arguments
        :rtype: :class:`list` of :class:`str` | None
        :raises: None
        """
        if not arguments:
            return None
        a = arguments.split(" :", 1)
        arglist = a[0].split()
        if len(a) == 2:
            arglist.append(a[1])
        return arglist