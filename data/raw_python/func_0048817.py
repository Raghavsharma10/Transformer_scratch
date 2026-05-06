def descriptor_for_symbol(self, symbol):
        """
        Given the symbol associated with the problem.
        Returns the :class:`~means.core.descriptors.Descriptor` associated with that symbol

        :param symbol: Symbol
        :type symbol: basestring|:class:`sympy.Symbol`
        :return:
        """
        if isinstance(symbol, basestring):
            symbol = sympy.Symbol(symbol)

        try:
            return self._descriptions_dict[symbol]
        except KeyError:
            raise KeyError("Symbol {0!r} not found in left-hand-side of the equations".format(symbol))