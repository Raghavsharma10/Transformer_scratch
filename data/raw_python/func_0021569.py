def run(self, lam, initial_values=None):
        '''Run the graph-fused logit lasso with a fixed lambda penalty.'''
        if initial_values is not None:
            if self.k == 0 and self.trails is not None:
                betas, zs, us = initial_values
            else:
                betas, us = initial_values
        else:
            if self.k == 0 and self.trails is not None:
                betas = [np.zeros(self.num_nodes, dtype='double') for _ in self.bins]
                zs = [np.zeros(self.breakpoints[-1], dtype='double') for _ in self.bins]
                us = [np.zeros(self.breakpoints[-1], dtype='double') for _ in self.bins]
            else:
                betas = [np.zeros(self.num_nodes, dtype='double') for _ in self.bins]
                us = [np.zeros(self.Dk.shape[0], dtype='double') for _ in self.bins]

        for j, (left, mid, right, trials, successes) in enumerate(self.bins):
            if self.bins_allowed is not None and j not in self.bins_allowed:
                continue

            if self.verbose > 2:
                print('\tBin #{0} [{1},{2},{3}]'.format(j, left, mid, right))
            # if self.verbose > 3:
            #     print 'Trials:\n{0}'.format(pretty_str(trials))
            #     print ''
            #     print 'Successes:\n{0}'.format(pretty_str(successes))
                
            beta = betas[j]
            u = us[j]

            if self.k == 0 and self.trails is not None:
                z = zs[j]
                # Run the graph-fused lasso algorithm
                self.graphfl(len(beta), trials, successes,
                             self.ntrails, self.trails, self.breakpoints,
                             lam, self.alpha, self.inflate,
                             self.max_steps, self.converge,
                             beta, z, u)
            else:
                # Run the graph trend filtering algorithm
                self.graphtf(len(beta), trials, successes, lam,
                                 self.Dk.shape[0], self.Dk.shape[1], self.Dk.nnz,
                                 self.Dk.row.astype('int32'), self.Dk.col.astype('int32'), self.Dk.data.astype('double'),
                                 self.max_steps, self.converge,
                                 beta, u)
                beta = np.clip(beta, 1e-12, 1-1e-12) # numerical stability
                betas[j] = -np.log(1./beta - 1.) # convert back to natural parameter form

        return (betas, zs, us) if self.k == 0 and self.trails is not None else (betas, us)