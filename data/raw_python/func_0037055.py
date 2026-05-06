def __computeDecomp(self):
        """
        Compute optimal dedomposition, each sub-domain has the
        same volume in index space.
        @return list if successful, empty list if not successful
        """
        primeNumbers = [getPrimeFactors(d) for d in self.globalDims]

        ns = [len(pns) for pns in primeNumbers]
        validDecomps = []
        self.validProcs = []
        for it in MultiArrayIter(ns):
            inds = it.getIndices()
            decomp = [primeNumbers[d][inds[d]] for d in range(self.ndims)]
            self.validProcs.append(reduce(operator.mul, decomp, 1))
            if reduce(operator.mul, decomp, 1) == self.nprocs:
                validDecomps.append(decomp)

        # sort and remove duplicates
        self.validProcs.sort()
        vprocs = []
        for vp in self.validProcs:
            if len(vprocs) == 0 or (len(vprocs) >= 1 and vp != vprocs[-1]):
                vprocs.append(vp)
        self.validProcs = vprocs

        if len(validDecomps) == 0:
            # no solution
            return

        # find the optimal decomp among all valid decomps
        minCost = float('inf')
        bestDecomp = validDecomps[0]
        for decomp in validDecomps:
            sizes = [self.globalDims[d]//decomp[d] for d in range(self.ndims)]
            volume = reduce(operator.mul, sizes, 1)
            surface = 0
            for d in range(self.ndims):
                surface += 2*reduce(operator.mul, sizes[:d], 1) * \
                    reduce(operator.mul, sizes[d+1:], 1)
            cost = surface / float(volume)
            if cost < minCost:
                bestDecomp = decomp
                minCost = cost
        self.decomp = bestDecomp

        # ok, we have a valid decomp, now build the sub-domain iterator
        self.mit = MultiArrayIter(self.decomp, rowMajor=self.rowMajor)

        # fill in the proc to index set map
        procId = 0
        self.proc2IndexSet = {}
        numCellsPerProc = [self.globalDims[d]//self.decomp[d]
                           for d in range(self.ndims)]
        for it in self.mit:
            nps = it.getIndices()
            self.proc2IndexSet[procId] = []
            for d in range(self.ndims):
                sbeg = nps[d]*numCellsPerProc[d]
                send = (nps[d] + 1)*numCellsPerProc[d]
                self.proc2IndexSet[procId].append(slice(sbeg, send))
            procId += 1