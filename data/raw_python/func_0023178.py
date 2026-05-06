def add_chain(self, var):
        """
        Create a new ChainFunction and attach to $var.
        """
        chain = FunctionChain(var, [])
        self._chains[var] = chain
        self[var] = chain