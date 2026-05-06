def download_and_calibrate(img_id=None, overwrite=False, recalibrate=False, **kwargs):
    """Download and calibrate one or more image ids, in parallel.

    Parameters
    ----------
    img_id : str or io.PathManager, optional
        If more than one item is in img_id, a parallel process is started
    overwrite: bool, optional
        If the pm.cubepath exists, this switch controls if it is being overwritten.
        Default: False
    """
    if isinstance(img_id, io.PathManager):
        pm = img_id
    else:
        # get a PathManager object that knows where your data is or should be
        logger.debug("Creating Pathmanager object")
        pm = io.PathManager(img_id)

    if not pm.raw_image.exists() or overwrite is True:
        logger.debug("Downloading file %s" % pm.img_id)
        download_file_id(pm.img_id)
        pm = io.PathManager(img_id)  # refresh, to get proper PDS version id.
    else:
        logger.info("Found ")

    if not (pm.cubepath.exists() and pm.undestriped.exists()) or overwrite is True:
        calib = pipeline.Calibrator(img_id, **kwargs)
        calib.standard_calib()
    else:
        print("All files exist. Use overwrite=True to redownload and calibrate.")