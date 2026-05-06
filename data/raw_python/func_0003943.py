def _rough_shake(self, x, normals, values, error):
        '''Take a robust, but not very efficient step towards the constraints.

           Arguments:
            | ``x`` -- The unknowns.
            | ``normals`` -- A numpy array with the gradients of the active
                             constraints. Each row is one gradient.
            | ``values`` -- A numpy array with the values of the constraint
                            functions.
            | ``error`` -- The square root of the constraint cost function.
        '''
        counter = 0
        while error > self.threshold and counter < self.max_iter:
            dxs = []
            for i in range(len(normals)):
                dx = -normals[i]*values[i]/np.dot(normals[i], normals[i])
                dxs.append(dx)
            dxs = np.array(dxs)
            dx = dxs[abs(values).argmax()]
            x = x+dx
            self.lock[:] = False
            normals, values, error = self._compute_equations(x)[:-1]
            counter += 1
        return x, normals, values, error