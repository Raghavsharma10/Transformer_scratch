def update(self, *args):
        """ Updates information about colors and their multiplicity in respective :class:`Multicolor` instance.

        By iterating over supplied arguments each of which should represent a color object, updates information about colors and their multiplicity in current :class:`Multicolor` instance.

        :param args: variable number of colors to add to currently existing multi colors data
        :type args: any hashable python object
        :return: ``None``, performs inplace changes to :attr:`Multicolor.multicolors` attribute
        """
        self.multicolors = self.multicolors + Counter(arg for arg in args)