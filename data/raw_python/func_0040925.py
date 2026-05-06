def filer(filelist):
    """
    Helper script that creates a dictionary of the stain name: /sequencepath/strain_name.extension)
    :param filelist: list of files to parse
    :return filedict: dictionary of stain name: /sequencepath/strain_name.extension
    """
    # Initialise the dictionary
    filedict = dict()
    for seqfile in filelist:
        # Split off the file extension and remove the path from the name
        strainname = os.path.splitext(os.path.basename(seqfile))[0]
        # Populate the dictionary
        filedict[strainname] = seqfile
    return filedict