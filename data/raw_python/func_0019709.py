def update(self, egg, permute=False, nperms=1000,
                 parallel=False):
        """
        In-place method that updates fingerprint with new data

        Parameters
        ----------
        egg : quail.Egg
            Data to update fingerprint
        Returns
        ----------
        None
        """

        # increment n
        self.n+=1

        next_weights = np.nanmean(_analyze_chunk(egg,
                          analysis=fingerprint_helper,
                          analysis_type='fingerprint',
                          pass_features=True,
                          permute=permute,
                          n_perms=nperms,
                          parallel=parallel).values, 0)

        if self.state is not None:

            # multiply states by n
            c = self.state*self.n

            # update state
            self.state = np.nansum(np.array([c, next_weights]), axis=0)/(self.n+1)

        else:

            self.state = next_weights

        # update the history
        self.history.append(next_weights)