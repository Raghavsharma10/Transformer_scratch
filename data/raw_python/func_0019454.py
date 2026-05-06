def remove_chain(self, chain=-1):
        """
        Removes a chain from ChainConsumer. Calling this will require any configurations set to be redone!

        Parameters
        ----------
        chain : int|str, list[str|int]
            The chain(s) to remove. You can pass in either the chain index, or the chain name, to remove it.
            By default removes the last chain added.

        Returns
        -------
        ChainConsumer
            Itself, to allow chaining calls.
        """
        if isinstance(chain, str) or isinstance(chain, int):
            chain = [chain]

        chain = sorted([i for c in chain for i in self._get_chain(c)])[::-1]
        assert len(chain) == len(list(set(chain))), "Error, you are trying to remove a chain more than once."

        for index in chain:
            del self.chains[index]

        seen = set()
        self._all_parameters = [p for c in self.chains for p in c.parameters if not (p in seen or seen.add(p))]

        # Need to reconfigure
        self._init_params()

        return self