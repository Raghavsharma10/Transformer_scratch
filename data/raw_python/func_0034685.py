def __left_merge(multicolor1, multicolor2):
        """ Updates first supplied :class:`Multicolor` instance with information from second supplied :class:`Multicolor` instance.

        First supplied instances attribute :attr:`Multicolor.multicolors` is updated with a help of ``+`` method of python Counter object.

        :param multicolor1: instance to update information in
        :type multicolor1: :class:`Multicolor`
        :param multicolor2: instance to use information for update from
        :type multicolor2: :class:`Multicolor`
        :return: updated first supplied :class:`Multicolor` instance
        :rtype: :class:`Multicolor`
        """
        multicolor1.multicolors = multicolor1.multicolors + multicolor2.multicolors
        return multicolor1