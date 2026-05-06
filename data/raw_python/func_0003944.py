def _fast_shake(self, x, normals, values, error):
        '''Take an efficient (not always robust) step towards the constraints.

           Arguments:
            | ``x`` -- The unknowns.
            | ``normals`` -- A numpy array with the gradients of the active
                             constraints. Each row is one gradient.
            | ``values`` -- A numpy array with the values of the constraint
                            functions.
            | ``error`` -- The square root of the constraint cost function.
        '''
        # filter out the degrees of freedom that do not feel the constraints.
        mask = (normals!=0).any(axis=0) > 0
        normals = normals[:,mask]
        # Take a step to lower the constraint cost function. If the step is too
        # large, it is reduced iteratively towards a small steepest descent
        # step. This is very similar to the Levenberg-Marquardt algorithm.
        # Singular Value decomposition is used to make this procedure
        # numerically more stable and efficient.
        U, S, Vt = np.linalg.svd(normals, full_matrices=False)
        rcond = None
        counter = 0
        while True:
            if rcond is None:
                rcond = 0.0
            elif rcond == 0.0:
                rcond = self.rcond1
            else:
                rcond *= 10
            # perform the least-norm correction
            Sinv = (S**2+rcond)
            if Sinv.max() == 0.0:
                continue
            Sinv = S/Sinv
            # compute the step
            dx = -np.dot(Vt.transpose(), np.dot(U.transpose(), values)*Sinv)
            new_x = x.copy()
            new_x[mask] += dx
            # try the step
            new_normals, new_values, new_error = self._compute_equations(new_x)[:-1]
            if new_error < 0.9*error:
                # Only if it decreases the constraint cost sufficiently, the
                # step is accepted. This routine is pointless of it converges
                # slowly.
                return new_x, new_normals, new_values, new_error
            elif abs(dx).sum() < self.threshold:
                # If the step becomes too small, then give up.
                break
            elif counter > self.max_iter:
                raise ConstraintError('Exceeded maximum number of shake iterations.')
            counter += 1