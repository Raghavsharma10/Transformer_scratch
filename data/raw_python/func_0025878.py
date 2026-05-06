def checkFITSFormat(filelist, ivmlist=None):
    """
    This code will check whether or not files are GEIS or WAIVER FITS and
    convert them to MEF if found. It also keeps the IVMLIST consistent with
    the input filelist, in the case that some inputs get dropped during
    the check/conversion.
    """
    if ivmlist is None:
        ivmlist = [None for l in filelist]

    sci_ivm = list(zip(filelist, ivmlist))

    removed_files, translated_names, newivmlist = convert2fits(sci_ivm)
    newfilelist, ivmlist = update_input(filelist, ivmlist, removed_files)

    if newfilelist == [] and translated_names == []:
        return [], []

    elif translated_names != []:
        newfilelist.extend(translated_names)
        ivmlist.extend(newivmlist)

    return newfilelist, ivmlist