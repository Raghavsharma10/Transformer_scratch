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
        {'D': 0, 'J': 13, 'V': 36, 'delDl': 2, 'delDr': 13, 'delJ': 10, 'delV': 5, 'insDJ': 6, 'insVD': 9}

        """

        recomb_events = {}
        recomb_events['V'] = self.CPV.searchsorted(np.random.random())

        #For 2D arrays make sure to take advantage of a mod expansion to find indicies
        DJ_choice = self.CPDJ.searchsorted(np.random.random())
        recomb_events['D'] = DJ_choice/self.num_J_genes
        recomb_events['J'] = DJ_choice % self.num_J_genes


        #Refer to the correct slices for the dependent distributions
        recomb_events['delV'] = self.given_V_CPdelV[recomb_events['V'], :].searchsorted(np.random.random())

        recomb_events['delJ'] = self.given_J_CPdelJ[recomb_events['J'], :].searchsorted(np.random.random())

        delDldelDr_choice = self.given_D_CPdelDldelDr[recomb_events['D'], :].searchsorted(np.random.random())

        recomb_events['delDl'] = delDldelDr_choice/self.num_delDr_poss
        recomb_events['delDr'] = delDldelDr_choice % self.num_delDr_poss

        recomb_events['insVD'] = self.CinsVD.searchsorted(np.random.random())
        recomb_events['insDJ'] = self.CinsDJ.searchsorted(np.random.random())

        return recomb_events