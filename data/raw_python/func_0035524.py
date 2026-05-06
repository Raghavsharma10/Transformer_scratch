def _calculate_correctedF3X4(self):
        '''Calculate `phi` based on the empirical `e_pw` values'''
        def F(phi):
            phi_reshape = phi.reshape((3, N_NT))
            functionList = []
            stop_frequency = []

            for x in range(N_STOP):
                codonFrequency = STOP_CODON_TO_NT_INDICES[x] * phi_reshape
                codonFrequency = scipy.prod(codonFrequency.sum(axis=1))
                stop_frequency.append(codonFrequency)
            C = scipy.sum(stop_frequency)

            for p in range(3):
                for w in range(N_NT):
                    s = 0
                    for x in range(N_STOP):
                        if STOP_CODON_TO_NT_INDICES[x][p][w] == 1:
                            s += stop_frequency[x]
                    functionList.append((phi_reshape[p][w] - s)/(1 - C)
                            - self.e_pw[p][w])
            return functionList

        phi = self.e_pw.copy().flatten()
        with scipy.errstate(invalid='ignore'):
            result = scipy.optimize.root(F, phi,
                    tol=1e-8)
            assert result.success, "Failed: {0}".format(result)
            return result.x.reshape((3, N_NT))