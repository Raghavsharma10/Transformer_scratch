def __create_nlinks(self, data, inds=None, boundary_penalties_fcn=None):
        """
        Compute nlinks grid from data shape information. For boundary penalties
        are data (intensities) values are used.

        ins: Default is None. Used for multiscale GC. This are indexes of
        multiscale pixels. Next example shows one superpixel witn index 2.
        inds = [
            [1 2 2],
            [3 2 2],
            [4 5 6]]

        boundary_penalties_fcn: is function with one argument - axis. It can
            it can be used for setting penalty weights between neighbooring
            pixels.

        """
        # use the gerneral graph algorithm
        # first, we construct the grid graph
        start = time.time()
        if inds is None:
            inds = np.arange(data.size).reshape(data.shape)
        # if not self.segparams['use_boundary_penalties'] and \
        #         boundary_penalties_fcn is None :
        if boundary_penalties_fcn is None:
            # This is faster for some specific format
            edgx = np.c_[inds[:, :, :-1].ravel(), inds[:, :, 1:].ravel()]
            edgy = np.c_[inds[:, :-1, :].ravel(), inds[:, 1:, :].ravel()]
            edgz = np.c_[inds[:-1, :, :].ravel(), inds[1:, :, :].ravel()]

        else:
            logger.info("use_boundary_penalties")

            bpw = self.segparams["boundary_penalties_weight"]

            bpa = boundary_penalties_fcn(2)
            # id1=inds[:, :, :-1].ravel()
            edgx = np.c_[
                inds[:, :, :-1].ravel(),
                inds[:, :, 1:].ravel(),
                # cc * np.ones(id1.shape)
                bpw * bpa[:, :, 1:].ravel(),
            ]

            bpa = boundary_penalties_fcn(1)
            # id1 =inds[:, 1:, :].ravel()
            edgy = np.c_[
                inds[:, :-1, :].ravel(),
                inds[:, 1:, :].ravel(),
                # cc * np.ones(id1.shape)]
                bpw * bpa[:, 1:, :].ravel(),
            ]

            bpa = boundary_penalties_fcn(0)
            # id1 = inds[1:, :, :].ravel()
            edgz = np.c_[
                inds[:-1, :, :].ravel(),
                inds[1:, :, :].ravel(),
                # cc * np.ones(id1.shape)]
                bpw * bpa[1:, :, :].ravel(),
            ]

        # import pdb; pdb.set_trace()
        edges = np.vstack([edgx, edgy, edgz]).astype(np.int32)
        # edges - seznam indexu hran, kteres spolu sousedi\
        elapsed = time.time() - start
        self.stats["_create_nlinks time"] = elapsed
        logger.info("__create nlinks time " + str(elapsed))
        return edges