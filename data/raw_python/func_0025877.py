def checkFiles(filelist,ivmlist = None):
    """
    - Converts waiver fits sciece and data quality files to MEF format
    - Converts GEIS science and data quality files to MEF format
    - Checks for stis association tables and splits them into single imsets
    - Removes files with EXPTIME=0 and the corresponding ivm files
    - Removes files with NGOODPIX == 0 (to exclude saturated images)
    - Removes files with missing PA_V3 keyword

    The list of science files should match the list of ivm files at the end.
    """
    toclose = False
    if isinstance(filelist[0], str):
        toclose = True
    newfilelist, ivmlist = checkFITSFormat(filelist, ivmlist)

    # check for STIS association files. This must be done before
    # the other checks in order to handle correctly stis
    # assoc files
    newfilelist, ivmlist = checkStisFiles(newfilelist, ivmlist)
    if newfilelist == []:
        return [], []
    removed_expt_files = check_exptime(newfilelist)

    newfilelist, ivmlist = update_input(newfilelist, ivmlist, removed_expt_files)
    if newfilelist == []:
        return [], []
    removed_ngood_files = checkNGOODPIX(newfilelist)
    newfilelist, ivmlist = update_input(newfilelist, ivmlist, removed_ngood_files)
    if newfilelist == []:
        return [], []

    removed_pav3_files = checkPA_V3(newfilelist)
    newfilelist, ivmlist = update_input(newfilelist, ivmlist, removed_pav3_files)

    newfilelist, ivmlist = update_input(newfilelist, ivmlist,[])

    if newfilelist == []:
        return [], []

    if toclose:
        newfilelist = [hdul.filename() for hdul in newfilelist]
    return newfilelist, ivmlist