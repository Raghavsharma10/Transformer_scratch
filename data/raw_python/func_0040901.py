def copy(self, klass=_x):
        """A new chain beginning with the current chain tokens and argument.
        """
        chain = super().copy()
        new_chain = klass(chain._args[0])
        new_chain._tokens = [[
            chain.compose, [], {},
        ]]
        return new_chain