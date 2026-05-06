def get_summary(self, squeeze=True, parameters=None, chains=None):
        """  Gets a summary of the marginalised parameter distributions.

        Parameters
        ----------
        squeeze : bool, optional
            Squeeze the summaries. If you only have one chain, squeeze will not return
            a length one list, just the single summary. If this is false, you will
            get a length one list.
        parameters : list[str], optional
            A list of parameters which to generate summaries for.
        chains : list[int|str], optional
            A list of the chains to get a summary of.

        Returns
        -------
        list of dictionaries
            One entry per chain, parameter bounds stored in dictionary with parameter as key
        """
        results = []
        if chains is None:
            chains = self.parent.chains
        else:
            if isinstance(chains, (int, str)):
                chains = [chains]
            chains = [self.parent.chains[i] for c in chains for i in self.parent._get_chain(c)]

        for chain in chains:
            res = {}
            params_to_find = parameters if parameters is not None else chain.parameters
            for p in params_to_find:
                if p not in chain.parameters:
                    continue
                summary = self.get_parameter_summary(chain, p)
                res[p] = summary
            results.append(res)
        if squeeze and len(results) == 1:
            return results[0]
        return results