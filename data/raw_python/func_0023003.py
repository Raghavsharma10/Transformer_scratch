def _neighbour_pixels(hp, ipix):
        """
        Returns all the pixels neighbours of ``ipix``
        """
        neigh_ipix = np.unique(hp.neighbours(ipix).ravel())
        # Remove negative pixel values returned by `~astropy_healpix.HEALPix.neighbours`
        return neigh_ipix[np.where(neigh_ipix >= 0)]