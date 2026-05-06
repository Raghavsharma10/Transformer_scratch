def find_best_candidate(self, elev_source_files=None):
        """
        Heuristically determines which tile should be recalculated based on
        updated edge information. Presently does not check if that tile is
        locked, which could lead to a parallel thread closing while one thread
        continues to process tiles.
        """
        self.fill_percent_done()
        i_b = np.argmax(self.percent_done.values())
        if self.percent_done.values()[i_b] <= 0:
            return None

        # check for ties
        I = np.array(self.percent_done.values()) == \
            self.percent_done.values()[i_b]
        if I.sum() == 1:
            pass  # no ties
        else:
            I2 = np.argmax(np.array(self.max_elev.values())[I])
            i_b = I.nonzero()[0][I2]

            # Make sure the apples are still apples
            assert(np.array(self.max_elev.keys())[I][I2]
                   == np.array(self.percent_done.keys())[I][I2])

        if elev_source_files is not None:
            fn = self.percent_done.keys()[i_b]
            lckfn = _get_lockfile_name(fn)
            if os.path.exists(lckfn):  # another process is working on it
                # Find a different Candidate
                i_alt = np.argsort(self.percent_done.values())[::-1]
                for i in i_alt:
                    fn = self.percent_done.keys()[i]
                    lckfn = _get_lockfile_name(fn)
                    if not os.path.exists(lckfn):
                        break
            # Get and return the index
            i_b = elev_source_files.index(fn)

        return i_b