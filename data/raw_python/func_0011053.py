def _generate_moment_matrix(self, n_vars, block_index, processed_entries,
                                monomialsA, monomialsB, ppt=False):
        """Generate the moment matrix of monomials.

        Arguments:
        n_vars -- current number of variables
        block_index -- current block index in the SDP matrix
        monomials -- |W_d| set of words of length up to the relaxation level
        """
        row_offset = 0
        if block_index > 0:
            for block_size in self.block_struct[0:block_index]:
                row_offset += block_size ** 2
        N = len(monomialsA)*len(monomialsB)
        func = partial(assemble_monomial_and_do_substitutions,
                       monomialsA=monomialsA, monomialsB=monomialsB, ppt=ppt,
                       substitutions=self.substitutions,
                       pure_substitution_rules=self.pure_substitution_rules)
        if self._parallel:
            pool = Pool()
            # This is just a guess and can be optimized
            chunksize = int(max(int(np.sqrt(len(monomialsA) * len(monomialsB) *
                                        len(monomialsA) / 2) / cpu_count()),
                                1))
        for rowA in range(len(monomialsA)):
            if self._parallel:
                iter_ = pool.map(func, [(rowA, columnA, rowB, columnB)
                                        for rowB in range(len(monomialsB))
                                        for columnA in range(rowA,
                                                             len(monomialsA))
                                        for columnB in range((rowA == columnA)*rowB,
                                                             len(monomialsB))],
                                 chunksize)
                print_criterion = processed_entries + len(iter_)
            else:
                iter_ = imap(func, [(rowA, columnA, rowB, columnB)
                                    for columnA in range(rowA, len(monomialsA))
                                    for rowB in range(len(monomialsB))
                                    for columnB in range((rowA == columnA)*rowB,
                                                         len(monomialsB))])
            for columnA, rowB, columnB, monomial in iter_:
                processed_entries += 1
                n_vars = self._push_monomial(monomial, n_vars,
                                             row_offset, rowA,
                                             columnA, N, rowB,
                                             columnB, len(monomialsB),
                                             prevent_substitutions=True)
                if self.verbose > 0 and (not self._parallel or
                                         processed_entries == self.n_vars or
                                         processed_entries == print_criterion):
                    percentage = processed_entries / self.n_vars
                    time_used = time.time()-self._time0
                    eta = (1.0 / percentage) * time_used - time_used
                    hours = int(eta/3600)
                    minutes = int((eta-3600*hours)/60)
                    seconds = eta-3600*hours-minutes*60

                    msg = ""
                    if self.verbose > 1 and self._parallel:
                        msg = ", working on block {:0} with {:0} processes with a chunksize of {:0d}"\
                              .format(block_index, cpu_count(),
                                      chunksize)
                    msg = "{:0} (done: {:.2%}, ETA {:02d}:{:02d}:{:03.1f}"\
                          .format(n_vars, percentage, hours, minutes, seconds) + \
                          msg
                    msg = "\r\x1b[KCurrent number of SDP variables: " + msg + ")"
                    sys.stdout.write(msg)
                    sys.stdout.flush()

        if self._parallel:
            pool.close()
            pool.join()
        if self.verbose > 0:
            sys.stdout.write("\r")
        return n_vars, block_index + 1, processed_entries