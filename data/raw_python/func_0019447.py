def get_max_posteriors(self, parameters=None, squeeze=True, chains=None):
        """  Gets the maximum posterior point in parameter space from the passed parameters.
        
        Requires the chains to have set `posterior` values.

        Parameters
        ----------
        parameters : str|list[str]
            The parameters to find
        squeeze : bool, optional
            Squeeze the summaries. If you only have one chain, squeeze will not return
            a length one list, just the single summary. If this is false, you will
            get a length one list.
        chains : list[int|str], optional
            A list of the chains to get a summary of.

        Returns
        -------
        list of two-tuples
            One entry per chain, two-tuple represents the max-likelihood coordinate
        """

        results = []
        if chains is None:
            chains = self.parent.chains
        else:
            if isinstance(chains, (int, str)):
                chains = [chains]
            chains = [self.parent.chains[i] for c in chains for i in self.parent._get_chain(c)]

        if isinstance(parameters, str):
            parameters = [parameters]

        for chain in chains:
            if chain.posterior_max_index is None:
                results.append(None)
                continue
            res = {}
            params_to_find = parameters if parameters is not None else chain.parameters
            for p in params_to_find:
                if p in chain.parameters:
                    res[p] = chain.posterior_max_params[p]
            results.append(res)

        if squeeze and len(results) == 1:
            return results[0]
        return results