def project(self, x, vector):
        '''Project a vector (gradient or direction) on the active constraints.

           Arguments:
            | ``x`` -- The unknowns.
            | ``vector`` -- A numpy array with a direction or a gradient.

           The return value is a gradient or direction, where the components
           that point away from the constraints are projected out. In case of
           half-open constraints, the projection is only active of the vector
           points into the infeasible region.
        '''
        scale = np.linalg.norm(vector)
        if scale == 0.0:
            return vector
        self.lock[:] = False
        normals, signs = self._compute_equations(x)[::3]
        if len(normals) == 0:
            return vector

        vector = vector/scale
        mask = signs == 0
        result = vector.copy()
        changed = True
        counter = 0
        while changed:
            changed = False
            y = np.dot(normals, result)
            for i, sign in enumerate(signs):
                if sign != 0:
                    if sign*y[i] < -self.threshold:
                        mask[i] = True
                        changed = True
                    elif mask[i] and np.dot(normals[i], result-vector) < 0:
                        mask[i] = False
                        changed = True

            if mask.any():
                normals_select = normals[mask]
                y = np.dot(normals_select, vector)
                U, S, Vt = np.linalg.svd(normals_select, full_matrices=False)
                if S.min() == 0.0:
                    Sinv = S/(S**2+self.rcond1)
                else:
                    Sinv = 1.0/S
                result = vector - np.dot(Vt.transpose(), np.dot(U.transpose(), y)*Sinv)
            else:
                result = vector.copy()

            if counter > self.max_iter:
                raise ConstraintError('Exceeded maximum number of shake iterations.')
            counter += 1

        return result*scale