def trivialInput(symbol):
    """
    Create a new L{IRichInput} implementation for the given input symbol.

    This creates a new type object and is intended to be used at module scope
    to define rich input types.  Generally, only one use per symbol should be
    required.  For example::

        Apple = trivialInput(Fruit.apple)

    @param symbol: A symbol from some state machine's input alphabet.

    @return: A new type object usable as a rich input for the given symbol.
    @rtype: L{type}
    """
    return implementer(IRichInput)(type(
            symbol.name.title(), (FancyStrMixin, object), {
                "symbol": _symbol(symbol),
                }))