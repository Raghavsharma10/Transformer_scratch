def convert2fits(sci_ivm):
    """
    Checks if a file is in WAIVER of GEIS format and converts it to MEF
    """
    removed_files = []
    translated_names = []
    newivmlist = []

    for file in sci_ivm:
        #find out what the input is
        # if science file is not found on disk, add it to removed_files for removal
        try:
            imgfits,imgtype = fileutil.isFits(file[0])
        except IOError:
            print("Warning:  File %s could not be found" %file[0])
            print("Warning:  Removing file %s from input list" %file[0])
            removed_files.append(file[0])
            continue

        # Check for existence of waiver FITS input, and quit if found.
        # Or should we print a warning and continue but not use that file
        if imgfits and imgtype == 'waiver':
            newfilename = waiver2mef(file[0], convert_dq=True)
            if newfilename is None:
                print("Removing file %s from input list - could not convert WAIVER format to MEF\n" %file[0])
                removed_files.append(file[0])
            else:
                removed_files.append(file[0])
                translated_names.append(newfilename)
                newivmlist.append(file[1])

        # If a GEIS image is provided as input, create a new MEF file with
        # a name generated using 'buildFITSName()'
        # Convert the corresponding data quality file if present
        if not imgfits:
            newfilename = geis2mef(file[0], convert_dq=True)
            if newfilename is None:
                print("Removing file %s from input list - could not convert GEIS format to MEF\n" %file[0])
                removed_files.append(file[0])
            else:
                removed_files.append(file[0])
                translated_names.append(newfilename)
                newivmlist.append(file[1])

    return removed_files, translated_names, newivmlist