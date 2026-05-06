def cues(self, rename_inhibitors=False):
        """
        Returns stimuli and inhibitors species of this experimental setup

        Parameters
        ----------
        rename_inhibitors : boolean
            If True, rename inhibitors with an ending 'i' as in MIDAS files.

        Returns
        -------
        list
            List of species names in order: first stimuli followed by inhibitors
        """
        if rename_inhibitors:
            return self.stimuli + [i+'i' for i in self.inhibitors]
        else:
            return self.stimuli + self.inhibitors