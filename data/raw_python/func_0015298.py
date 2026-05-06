def method(self, symbol_name):
        """
        A decorator - adds the decorated method to symbol 'symbol_name'
        """

        s = self.symbol(symbol_name)

        def bind(fn):
            setattr(s, fn.__name__, fn)
        return bind