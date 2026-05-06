def parseinput(inputlist,outputname=None, atfile=None):
    """
    Recursively parse user input based upon the irafglob
    program and construct a list of files that need to be processed.
    This program addresses the following deficiencies of the irafglob program::

       parseinput can extract filenames from association tables

    Returns
    -------
    This program will return a list of input files that will need to
    be processed in addition to the name of any outfiles specified in
    an association table.

    Parameters
    ----------
    inputlist - string
        specification of input files using either wild-cards, @-file or
        comma-separated list of filenames

    outputname - string
        desired name for output product to be created from the input files

    atfile - object
        function to use in interpreting the @-file columns that gets passed to irafglob

    Returns
    -------
    files - list of strings
        names of output files to be processed
    newoutputname - string
        name of output file to be created.

    See Also
    --------
    stsci.tools.irafglob

    """

    # Initalize some variables
    files = [] # list used to store names of input files
    newoutputname = outputname # Outputname returned to calling program.
                               # The value of outputname is only changed
                               # if it had a value of 'None' on input.


    # We can use irafglob to parse the input.  If the input wasn't
    # an association table, it needs to be either a wildcard, '@' file,
    # or comma seperated list.
    files = irafglob(inputlist, atfile=atfile)

    # Now that we have expanded the inputlist into a python list
    # containing the list of input files, it is necessary to examine
    # each of the files to make sure none of them are association tables.
    #
    # If an association table is found, the entries should be read
    # Determine if the input is an association table
    for file in files:
        if (checkASN(file) == True):
            # Create a list to store the files extracted from the
            # association tiable
            assoclist = []

            # The input is an association table
            try:
                # Open the association table
                assocdict = readASNTable(file, None, prodonly=False)
            except:
                errorstr  = "###################################\n"
                errorstr += "#                                 #\n"
                errorstr += "# UNABLE TO READ ASSOCIATION FILE,#\n"
                errorstr +=  str(file)+'\n'
                errorstr += "# DURING FILE PARSING.            #\n"
                errorstr += "#                                 #\n"
                errorstr += "# Please determine if the file is #\n"
                errorstr += "# in the current directory and    #\n"
                errorstr += "# that it has been properly       #\n"
                errorstr += "# formatted.                      #\n"
                errorstr += "#                                 #\n"
                errorstr += "# This error message is being     #\n"
                errorstr += "# generated from within the       #\n"
                errorstr += "# parseinput.py module.           #\n"
                errorstr += "#                                 #\n"
                errorstr += "###################################\n"
                raise ValueError(errorstr)

            # Extract the output name from the association table if None
            # was provided on input.
            if outputname is None:
                newoutputname = assocdict['output']

            # Loop over the association dictionary to extract the input
            # file names.
            for f in assocdict['order']:
                assoclist.append(fileutil.buildRootname(f))

            # Remove the name of the association table from the list of files
            files.remove(file)
            # Append the list of filenames generated from the association table
            # to the master list of input files.
            files.extend(assoclist)

    # Return the list of the input files and the output name if provided in an association.
    return files, newoutputname