def check_background(qpi):
    """Check QPimage background data

    Parameters
    ----------
    qpi: qpimage.core.QPImage

    Raises
    ------
    IntegrityCheckError
        if the check fails
    """
    for imdat in [qpi._amp, qpi._pha]:
        try:
            fit, attrs = imdat.get_bg(key="fit", ret_attrs=True)
        except KeyError:
            # No bg correction performed
            pass
        else:
            kwargs = dict(attrs)
            # check if we have a user-defined mask image
            binkey = "estimate_bg_from_mask"
            if binkey in imdat.h5:
                kwargs["from_mask"] = imdat.h5[binkey][:]
            else:
                kwargs["from_mask"] = None
            # compute background correction
            with h5py.File("check.h5",
                           driver="core",
                           backing_store=False) as h5:
                # imdat.__class__ is "Amplitude" or "Phase"
                testimdat = imdat.__class__(h5)
                testimdat["raw"] = imdat.raw
                # Set experimental bg data if given
                try:
                    bg = imdat.get_bg("data")
                except KeyError:
                    pass
                else:
                    testimdat.set_bg(bg, key="data")
                # fit bg
                testimdat.estimate_bg(**kwargs)
                # compare
                if not np.allclose(testimdat.get_bg(key="fit"), fit):
                    msg = "Wrong estimated (fitted) background!"
                    raise IntegrityCheckError(msg)