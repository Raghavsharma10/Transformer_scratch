def run_parallel(self, para_func):
        """Run parallel calulation

        This will run the parallel calculation on self.num_processors.

        Args:
            para_func (obj): Function object to be used in parallel.

        Returns:
            (dict): Dictionary with parallel results.

        """
        if self.timer:
            start_timer = time.time()

        # for testing
        # check = parallel_snr_func(*self.args[10])
        # import pdb
        # pdb.set_trace()

        with mp.Pool(self.num_processors) as pool:
            print('start pool with {} processors: {} total processes.\n'.format(
                    self.num_processors, len(self.args)))

            results = [pool.apply_async(para_func, arg) for arg in self.args]
            out = [r.get() for r in results]
            out = {key: np.concatenate([out_i[key] for out_i in out]) for key in out[0].keys()}
        if self.timer:
            print("SNR calculation time:", time.time()-start_timer)
        return out