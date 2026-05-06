def _to_json(uniq):
        """
        Serializes a MOC to the JSON format.

        Parameters
        ----------
        uniq : `~numpy.ndarray`
            The array of HEALPix cells representing the MOC to serialize.

        Returns
        -------
        result_json : {str : [int]}
            A dictionary of HEALPix cell lists indexed by their depth.
        """
        result_json = {}

        depth, ipix = utils.uniq2orderipix(uniq)
        min_depth = np.min(depth[0])
        max_depth = np.max(depth[-1])

        for d in range(min_depth, max_depth+1):
            pix_index = np.where(depth == d)[0]
            if pix_index.size:
                # there are pixels belonging to the current order
                ipix_depth = ipix[pix_index]
                result_json[str(d)] = ipix_depth.tolist()

        return result_json