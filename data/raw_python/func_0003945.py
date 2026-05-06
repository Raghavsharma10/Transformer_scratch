def free_shake(self, x):
        '''Brings unknowns to the constraints.

           Arguments:
            | ``x`` -- The unknowns.
        '''
        self.lock[:] = False
        normals, values, error = self._compute_equations(x)[:-1]
        counter = 0
        while True:
            if error <= self.threshold:
                break
            # try a well-behaved move to the constrains
            result = self._fast_shake(x, normals, values, error)
            counter += 1
            if result is not None:
                x, normals, values, error = result
            else:
                # well-behaved move is too slow.
                # do a cumbersome move to satisfy constraints approximately.
                x, normals, values, error = self._rough_shake(x, normals, values, error)
                counter += 1
            # When too many iterations are required, just give up.
            if counter > self.max_iter:
                raise ConstraintError('Exceeded maximum number of shake iterations.')
        return x, counter, len(values)