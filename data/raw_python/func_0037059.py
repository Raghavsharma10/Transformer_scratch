def apply(self, localArray):
        """
        Apply stencil to data
        @param localArray local array
        @return new array on local proc
        """

        # input dist array
        inp = daZeros(localArray.shape, localArray.dtype)
        inp[...] = localArray
        inp.setComm(self.comm)

        # output array
        out = numpy.zeros(localArray.shape, localArray.dtype)

        # expose the dist array windows
        for disp, dpi in self.dpis.items():

            srcs = dpi['srcs']
            remoteWinIds = dpi['remoteWinIds']
            numParts = len(srcs)
            for i in range(numParts):
                inp.expose(srcs[i], winID=remoteWinIds[i])

        # apply the stencil
        for disp, weight in self.stencil.items():

            dpi = self.dpis[disp]

            dpi = self.dpis[disp]

            srcs = dpi['srcs']
            dsts = dpi['dsts']
            remoteRanks = dpi['remoteRanks']
            remoteWinIds = dpi['remoteWinIds']
            numParts = len(srcs)
            for i in range(numParts):
                srcSlce = srcs[i]
                dstSlce = dsts[i]
                remoteRank = remoteRanks[i]
                remoteWinId = remoteWinIds[i]

                # now apply the stencil
                if remoteRank == self.myRank:
                    # local updates
                    out[dstSlce] += weight*inp[srcSlce]
                else:
                    # remote fetch
                    out[dstSlce] += weight*inp.getData(remoteRank, remoteWinId)

        # some implementations require this
        inp.free()

        return out