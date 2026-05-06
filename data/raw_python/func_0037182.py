def apply(self, localArray):
        """
        Apply Laplacian stencil to data
        @param localArray local array
        @return new array on local proc
        """

        # input dist array
        inp = gdaZeros(localArray.shape, localArray.dtype, numGhosts=1)
        # output array
        out = numpy.zeros(localArray.shape, localArray.dtype)

        # no displacement term
        weight = self.stencil[self.zeros]
        out[...] += weight * localArray

        for disp in self.srcLocalDomains:

            weight = self.stencil[disp]

            # no communication required here
            srcDom = self.srcLocalDomains[disp]
            dstDom = self.dstLocalDomains[disp]

            out[dstDom] += weight * localArray[srcDom]

            #
            # now the part that requires communication
            #

            # set the ghost values
            srcSlab = self.srcSlab[disp]
            # copy
            inp[srcSlab] = localArray[srcSlab]

            # send over to local process
            dstSlab = self.dstSlab[disp]
            winId = self.winIds[disp]
            rk = self.neighRk[disp]

            # remote fetch
            out[dstSlab] += weight * inp.getData(rk, winId)

        # some implementations require this
        inp.free()

        return out