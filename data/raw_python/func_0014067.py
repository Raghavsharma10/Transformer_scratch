def choices(cls, blank=False):
        """ Choices for Enum
        :return: List of tuples (<value>, <human-readable value>)
        :rtype: list
        """
        choices = sorted([(key, value) for key, value in cls.values.items()], key=lambda x: x[0])
        if blank:
            choices.insert(0, ('', Enum.Value('', None, '', cls)))
        return choices