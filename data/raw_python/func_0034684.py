def __merge(cls, *multicolors):
        """ Produces a new :class:`Multicolor` object resulting from gathering information from all supplied :class:`Multicolor` instances.

        New :class:`Multicolor` is created and its :attr:`Multicolor.multicolors` attribute is updated with similar attributes of supplied :class:`Multicolor` objects.

        Accounts for subclassing.

        :param multicolors: variable number of :class:`Multicolor` objects
        :type multicolors: :class:`Multicolor`
        :return: object containing gathered information from all supplied arguments
        :rtype: :class:`Multicolor`
        """
        result = cls()
        for multicolor in multicolors:
            result.multicolors = result.multicolors + multicolor.multicolors
        return result