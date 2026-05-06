def _to_str(uniq):
        """
        Serializes a MOC to the STRING format.

        HEALPix cells are separated by a comma. The HEALPix cell at order 0 and number 10 is encoded
        by the string: "0/10", the first digit representing the depth and the second the HEALPix cell number
        for this depth. HEALPix cells next to each other within a specific depth can be expressed as a range and 
        therefore written like that: "12/10-150". This encodes the list of HEALPix cells from 10 to 150 at the
        depth 12.

        Parameters
        ----------
        uniq : `~numpy.ndarray`
            The array of HEALPix cells representing the MOC to serialize.

        Returns
        -------
        result : str
            The serialized MOC.
        """
        def write_cells(serial, a, b, sep=''):
            if a == b:
                serial += '{0}{1}'.format(a, sep)
            else:
                serial += '{0}-{1}{2}'.format(a, b, sep)
            return serial

        res = ''

        if uniq.size == 0: 
            return res

        depth, ipixels = utils.uniq2orderipix(uniq)
        min_depth = np.min(depth[0])
        max_depth = np.max(depth[-1])

        for d in range(min_depth, max_depth+1):
            pix_index = np.where(depth == d)[0]

            if pix_index.size > 0:
                # Serialize the depth followed by a slash
                res += '{0}/'.format(d)

                # Retrieve the pixel(s) for this depth
                ipix_depth = ipixels[pix_index]
                if ipix_depth.size == 1:
                    # If there is only one pixel we serialize it and
                    # go to the next depth
                    res = write_cells(res, ipix_depth[0], ipix_depth[0])
                else:
                    # Sort them in case there are several
                    ipix_depth = np.sort(ipix_depth)

                    beg_range = ipix_depth[0]
                    last_range = beg_range

                    # Loop over the sorted pixels by tracking the lower bound of
                    # the current range and the last pixel.
                    for ipix in ipix_depth[1:]:
                        # If the current pixel does not follow the previous one
                        # then we can end a range and serializes it
                        if ipix > last_range + 1:
                            res = write_cells(res, beg_range, last_range, sep=',')
                            # The current pixel is the beginning of a new range
                            beg_range = ipix

                        last_range = ipix

                    # Write the last range
                    res = write_cells(res, beg_range, last_range)

                # Add a ' ' separator before writing serializing the pixels of the next depth
                res += ' '

        # Remove the last ' ' character
        res = res[:-1]

        return res