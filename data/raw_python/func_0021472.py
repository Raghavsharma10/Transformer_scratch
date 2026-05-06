def choose_random_recomb_events(self):
        """Sample the genomic model for VDJ recombination events.

        Returns
        -------
        recomb_events : dict
            Dictionary of the VDJ recombination events. These are
            integers determining gene choice, deletions, and number of insertions.

        Example
        --------
        >>> sequence_generation.choose_random_recomb_events()
        {'J': 13, 'V': 36, 'delJ': 10, 'delV': 5, 'insVJ': 3}

        """

        recomb_events = {}

        #For 2D arrays make sure to take advantage of a mod expansion to find indicies
        VJ_choice = self.CPVJ.searchsorted(np.random.random())
        recomb_events['V'] = VJ_choice/self.num_J_genes
        recomb_events['J'] = VJ_choice % self.num_J_genes


        #Refer to the correct slices for the dependent distributions
        recomb_events['delV'] = self.given_V_CPdelV[recomb_events['V'], :].searchsorted(np.random.random())

        recomb_events['delJ'] = self.given_J_CPdelJ[recomb_events['J'], :].searchsorted(np.random.random())
        recomb_events['insVJ'] = self.CPinsVJ.searchsorted(np.random.random())

        return recomb_events