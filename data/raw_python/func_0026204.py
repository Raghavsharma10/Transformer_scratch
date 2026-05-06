def _verify(waiveredHdul):
    """
        Verify that the input HDUList is for a waivered FITS file.

        Parameters:

           waiveredHdul     HDUList object to be verified

        Returns: None

        Exceptions:

           ValueError       Input HDUList is not for a waivered FITS file
    """

    if len(waiveredHdul) == 2:
        #
        # There must be exactly 2 HDU's
        #
        if waiveredHdul[0].header['NAXIS'] > 0:
            #
            # The Primary HDU must have some data
            #
            if isinstance(waiveredHdul[1], fits.TableHDU):
                #
                # The Alternate HDU must be a TableHDU
                #
                if waiveredHdul[0].data.shape[0] == \
                   waiveredHdul[1].data.shape[0] or \
                   waiveredHdul[1].data.shape[0] == 1:
                    #
                    # The number of arrays in the Primary HDU must match
                    # the number of rows in the TableHDU.  This includes
                    # the case where there is only a single array and row.
                    #
                    return
    #
    # Not a valid waivered Fits file
    #
    raise ValueError("Input object does not represent a valid waivered" + \
                      " FITS file")