def __delete(self, multicolor):
        """ Reduces information :class:`Multicolor` attribute by iterating over supplied colors data.

        In case supplied argument is a :class:`Multicolor` instance, multi-color specific information to de deleted is set to its :attr:`Multicolor.multicolors`.
        In other cases multi-color specific information to de deleted is obtained from iterating over the argument.

        Colors and their multiplicity is reduces with a help of ``-`` method of python Counter object.

        :param multicolor: information about colors to be deleted from :class:`Multicolor` object
        :type multicolor: any iterable with colors object as entries or :class:`Multicolor`
        :return: ``None``, performs inplace changes
        """
        if isinstance(multicolor, Multicolor):
            to_delete = multicolor.multicolors
        else:
            to_delete = Counter(color for color in multicolor)
        self.multicolors = self.multicolors - to_delete