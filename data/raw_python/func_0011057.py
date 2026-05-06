def __process_inequalities(self, block_index):
        """Generate localizing matrices

        Arguments:
        inequalities -- list of inequality constraints
        monomials    -- localizing monomials
        block_index -- the current block index in constraint matrices of the
                       SDP relaxation
        """
        initial_block_index = block_index
        row_offsets = [0]
        for block, block_size in enumerate(self.block_struct):
            row_offsets.append(row_offsets[block] + block_size ** 2)

        if self._parallel:
            pool = Pool()
        for k, ineq in enumerate(self.constraints):
            block_index += 1
            monomials = self.localizing_monomial_sets[block_index -
                                                      initial_block_index-1]
            lm = len(monomials)
            if isinstance(ineq, str):
                self.__parse_expression(ineq, row_offsets[block_index-1])
                continue
            if ineq.is_Relational:
                ineq = convert_relational(ineq)
            func = partial(moment_of_entry, monomials=monomials, ineq=ineq,
                           substitutions=self.substitutions)
            if self._parallel and lm > 1:
                chunksize = max(int(np.sqrt(lm*lm/2) /
                                    cpu_count()), 1)
                iter_ = pool.map(func, ([row, column] for row in range(lm)
                                        for column in range(row, lm)),
                                 chunksize)
            else:
                iter_ = imap(func, ([row, column] for row in range(lm)
                                    for column in range(row, lm)))
            if block_index > self.constraint_starting_block + \
                    self._n_inequalities and lm > 1:
                is_equality = True
            else:
                is_equality = False
            for row, column, polynomial in iter_:
                if is_equality:
                    row, column = 0, 0
                self.__push_facvar_sparse(polynomial, block_index,
                                          row_offsets[block_index-1],
                                          row, column)
                if is_equality:
                    block_index += 1
            if is_equality:
                block_index -= 1
            if self.verbose > 0:
                sys.stdout.write("\r\x1b[KProcessing %d/%d constraints..." %
                                 (k+1, len(self.constraints)))
                sys.stdout.flush()
        if self._parallel:
            pool.close()
            pool.join()

        if self.verbose > 0:
            sys.stdout.write("\n")
        return block_index