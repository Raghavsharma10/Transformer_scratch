def spillover(self, quadrant=1, neighbors_on=False):
        """
        Detect spillover locations for diffusion in LISA Markov.

        Parameters
        ----------
        quadrant     : int
                       which quadrant in the scatterplot should form the core
                       of a cluster.
        neighbors_on : binary
                       If false, then only the 1st order neighbors of a core
                       location are included in the cluster.
                       If true, neighbors of cluster core 1st order neighbors
                       are included in the cluster.

        Returns
        -------
        results      : dictionary
                       two keys - values pairs:
                       'components' - array (n, t)
                       values are integer ids (starting at 1) indicating which
                       component/cluster observation i in period t belonged to.
                       'spillover' - array (n, t-1)
                       binary values indicating if the location was a
                       spill-over location that became a new member of a
                       previously existing cluster.

        Examples
        --------
        >>> import libpysal
        >>> from giddy.markov import LISA_Markov
        >>> f = libpysal.io.open(libpysal.examples.get_path("usjoin.csv"))
        >>> years = list(range(1929, 2010))
        >>> pci = np.array([f.by_col[str(y)] for y in years]).transpose()
        >>> w = libpysal.io.open(libpysal.examples.get_path("states48.gal")).read()
        >>> np.random.seed(10)
        >>> lm_random = LISA_Markov(pci, w, permutations=99)
        >>> r = lm_random.spillover()
        >>> (r['components'][:, 12] > 0).sum()
        17
        >>> (r['components'][:, 13]>0).sum()
        23
        >>> (r['spill_over'][:,12]>0).sum()
        6

        Including neighbors of core neighbors
        >>> rn = lm_random.spillover(neighbors_on=True)
        >>> (rn['components'][:, 12] > 0).sum()
        26
        >>> (rn["components"][:, 13] > 0).sum()
        34
        >>> (rn["spill_over"][:, 12] > 0).sum()
        8

        """
        n, k = self.q.shape
        if self.permutations:
            spill_over = np.zeros((n, k - 1))
            components = np.zeros((n, k))
            i2id = {}  # handle string keys
            for key in list(self.w.neighbors.keys()):
                idx = self.w.id2i[key]
                i2id[idx] = key
            sig_lisas = (self.q == quadrant) \
                * (self.p_values <= self.significance_level)
            sig_ids = [np.nonzero(
                sig_lisas[:, i])[0].tolist() for i in range(k)]

            neighbors = self.w.neighbors
            for t in range(k - 1):
                s1 = sig_ids[t]
                s2 = sig_ids[t + 1]
                g1 = Graph(undirected=True)
                for i in s1:
                    for neighbor in neighbors[i2id[i]]:
                        g1.add_edge(i2id[i], neighbor, 1.0)
                        if neighbors_on:
                            for nn in neighbors[neighbor]:
                                g1.add_edge(neighbor, nn, 1.0)
                components1 = g1.connected_components(op=gt)
                components1 = [list(c.nodes) for c in components1]
                g2 = Graph(undirected=True)
                for i in s2:
                    for neighbor in neighbors[i2id[i]]:
                        g2.add_edge(i2id[i], neighbor, 1.0)
                        if neighbors_on:
                            for nn in neighbors[neighbor]:
                                g2.add_edge(neighbor, nn, 1.0)
                components2 = g2.connected_components(op=gt)
                components2 = [list(c.nodes) for c in components2]
                c2 = []
                c1 = []
                for c in components2:
                    c2.extend(c)
                for c in components1:
                    c1.extend(c)

                new_ids = [j for j in c2 if j not in c1]
                spill_ids = []
                for j in new_ids:
                    # find j's component in period 2
                    cj = [c for c in components2 if j in c][0]
                    # for members of j's component in period 2, check if they
                    # belonged to any components in period 1
                    for i in cj:
                        if i in c1:
                            spill_ids.append(j)
                            break
                for spill_id in spill_ids:
                    id = self.w.id2i[spill_id]
                    spill_over[id, t] = 1
                for c, component in enumerate(components1):
                    for i in component:
                        ii = self.w.id2i[i]
                        components[ii, t] = c + 1
            results = {}
            results['components'] = components
            results['spill_over'] = spill_over
            return results

        else:
            return None