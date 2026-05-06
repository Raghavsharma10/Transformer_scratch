def _compute_equations(self, x, verbose=False):
        '''Compute the values and the normals (gradients) of active constraints.

           Arguments:
            | ``x`` -- The unknowns.
        '''
        # compute the error and the normals.
        normals = []
        values = []
        signs = []
        error = 0.0
        if verbose:
            print()
            print(' '.join('% 10.3e' % val for val in x), end=' ')
            active_str = ''
        for i, (sign, equation) in enumerate(self.equations):
            value, normal = equation(x)
            if (i < len(self.lock) and self.lock[i]) or \
               (sign==-1 and value > -self.threshold) or \
               (sign==0) or (sign==1 and value < self.threshold):
                values.append(value)
                normals.append(normal)
                signs.append(sign)
                error += value**2
                if verbose:
                    active_str += 'X'
                if i < len(self.lock):
                    self.lock[i] = True
            elif verbose:
                active_str += '-'
        error = np.sqrt(error)
        normals = np.array(normals, float)
        values = np.array(values, float)
        signs = np.array(signs, int)
        if verbose:
            print('[%s]' % active_str, end=' ')
            if error < self.threshold:
                print('OK')
            else:
                print('%.5e' % error)
        return normals, values, error, signs