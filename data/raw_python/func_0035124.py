def prep_parallel(self, binary_args, other_args):
        """Prepare the parallel calculations

        Prepares the arguments to be run in parallel.
        It will divide up arrays according to num_splits.

        Args:
            binary_args (list): List of binary arguments for input into the SNR function.
            other_args (tuple of obj): tuple of other args for input into parallel snr function.

        """
        if self.length < 100:
            raise Exception("Run this across 1 processor by setting num_processors kwarg to None.")
        if self.num_processors == -1:
            self.num_processors = mp.cpu_count()

        split_val = int(np.ceil(self.length/self.num_splits))
        split_inds = [self.num_splits*i for i in np.arange(1, split_val)]

        inds_split_all = np.split(np.arange(self.length), split_inds)

        self.args = []
        for i, ind_split in enumerate(inds_split_all):
            trans_args = []
            for arg in binary_args:
                try:
                    trans_args.append(arg[ind_split])
                except TypeError:
                    trans_args.append(arg)

            self.args.append((i, tuple(trans_args)) + other_args)
        return