def check_for_empty_defaults(status):
    """
    Method to check for empty roles structure.

    When a role is created using ansible-galaxy it creates a default
    scaffolding structure. Best practice dictates that if any of these are not
    used then they should be removed. For example a bare main.yml with the
    following string is created for a 'defaults' for a role called 'myrole':

    ---
    defaults file for myrole

    This should be removed.

    Args:
        status (list): list of pre-receive check failures to eventually print
                       to the user
    Returns:
       status list of current pre-redeive check failures. Might be an empty
       list.
    """

    dirs_to_check = ('./vars', './handlers', './defaults', './tasks')

    for dirpath, dirname, filename in os.walk('.'):

        if dirpath == './files' or dirpath == "./templates":
            if not any([dirname, filename]):
                status.append("There are no files in the {0} directory. please"
                              " remove directory".format(dirpath))

        if dirpath in dirs_to_check:
            try:
                joined_filename = os.path.join(dirpath, 'main.yml')
                with open(joined_filename, 'r') as f:
                    # try to match:
                    # ---
                    # (tasks|vars|defaults) file for myrole
                    #
                    if re.match(r'^---\n# \S+ file for \S+\n$', f.read()):
                        status.append("Empty file, please remove file and "
                                      "directory: {0}".format(joined_filename))
            except IOError:
                # Can't find a main.yml - but this could be legitimate
                pass

    return status