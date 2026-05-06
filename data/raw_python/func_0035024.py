def _computePartialLikelihoods(self):
        """Update `L`, `dL`, `dL_dt`."""
        for n in range(self.ntips, self.nnodes):
            ni = n - self.ntips # internal node number
            nright = self.rdescend[ni]
            nleft = self.ldescend[ni]
            if nright < self.ntips:
                istipr = True
            else:
                istipr = False
            if nleft < self.ntips:
                istipl = True
            else:
                istipl = False
            tright = self.t[nright]
            tleft = self.t[nleft]
            self.L[n] = scipy.ndarray(self._Lshape, dtype='float')
            if self.dparamscurrent:
                for param in self._paramlist_PartialLikelihoods:
                    self.dL[param][n] = scipy.ndarray(self._dLshape[param],
                            dtype='float')
            if self.dtcurrent:
                for n2 in self.dL_dt.keys():
                    self.dL_dt[n2][n] = scipy.zeros(self._Lshape, dtype='float')
            for k in self._catindices:
                if istipr:
                    Mright = MLright = self._M(k, tright,
                            self.tips[nright], self.gaps[nright])
                else:
                    Mright = self._M(k, tright)
                    MLright = broadcastMatrixVectorMultiply(Mright,
                            self.L[nright][k])
                if istipl:
                    Mleft = MLleft = self._M(k, tleft,
                            self.tips[nleft], self.gaps[nleft])
                else:
                    Mleft = self._M(k, tleft)
                    MLleft = broadcastMatrixVectorMultiply(Mleft,
                            self.L[nleft][k])
                self.L[n][k] = MLright * MLleft

                if self.dtcurrent:
                    for (tx, Mx, nx, MLxother, istipx) in [
                            (tright, Mright, nright, MLleft, istipr),
                            (tleft, Mleft, nleft, MLright, istipl)]:
                        if istipx:
                            tipsx = self.tips[nx]
                            gapsx = self.gaps[nx]
                        else:
                            tipsx = gapsx = None
                        dM_dt = self._dM(k, tx, 't', Mx, tipsx, gapsx)
                        if istipx:
                            LdM_dt = dM_dt
                        else:
                            LdM_dt = broadcastMatrixVectorMultiply(
                                    dM_dt, self.L[nx][k])
                        self.dL_dt[nx][n][k] = LdM_dt * MLxother
                        for ndx in self.descendants[nx]:
                            self.dL_dt[ndx][n][k] = broadcastMatrixVectorMultiply(
                                    Mx, self.dL_dt[ndx][nx][k]) * MLxother

                if self.dparamscurrent:
                    for param in self._paramlist_PartialLikelihoods:
                        if istipr:
                            dMright = self._dM(k, tright, param, Mright,
                                    self.tips[nright], self.gaps[nright])
                        else:
                            dMright = self._dM(k, tright, param, Mright)
                        if istipl:
                            dMleft = self._dM(k, tleft, param, Mleft,
                                    self.tips[nleft], self.gaps[nleft])
                        else:
                            dMleft = self._dM(k, tleft, param, Mleft)
                        for j in self._sub_index_param(param):
                            if istipr:
                                dMLright = dMright[j]
                                MdLright = 0
                            else:
                                dMLright = broadcastMatrixVectorMultiply(
                                        dMright[j], self.L[nright][k])
                                MdLright = broadcastMatrixVectorMultiply(
                                        Mright, self.dL[param][nright][k][j])
                            if istipl:
                                dMLleft = dMleft[j]
                                MdLleft = 0
                            else:
                                dMLleft = broadcastMatrixVectorMultiply(
                                        dMleft[j], self.L[nleft][k])
                                MdLleft = broadcastMatrixVectorMultiply(
                                        Mleft, self.dL[param][nleft][k][j])
                            self.dL[param][n][k][j] = ((dMLright + MdLright)
                                    * MLleft + MLright * (dMLleft + MdLleft))

            if ni > 0 and ni % self.underflowfreq == 0:
                # rescale by same amount for each category k
                scale = scipy.amax(scipy.array([scipy.amax(self.L[n][k],
                        axis=1) for k in self._catindices]), axis=0)
                assert scale.shape == (self.nsites,)
                self.underflowlogscale += scipy.log(scale)
                for k in self._catindices:
                    self.L[n][k] /= scale[:, scipy.newaxis]
                    if self.dtcurrent:
                        for n2 in self.dL_dt.keys():
                            self.dL_dt[n2][n][k] /= scale[:, scipy.newaxis]
                    if self.dparamscurrent:
                        for param in self._paramlist_PartialLikelihoods:
                            for j in self._sub_index_param(param):
                                self.dL[param][n][k][j] /= scale[:, scipy.newaxis]

            # free unneeded memory by deleting already used values
            for ntodel in [nright, nleft]:
                if ntodel in self.L:
                    del self.L[ntodel]
                if self.dparamscurrent:
                    for param in self._paramlist_PartialLikelihoods:
                        if ntodel in self.dL[param]:
                            del self.dL[param][ntodel]
                if self.dtcurrent:
                    for n2 in self.dL_dt.keys():
                        if ntodel in self.dL_dt[n2]:
                            del self.dL_dt[n2][ntodel]