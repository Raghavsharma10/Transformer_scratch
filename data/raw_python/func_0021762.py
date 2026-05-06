def call(command, silent=False):
    """ Runs a bash command safely, with shell=false, catches any non-zero
        return codes.  Raises slightly modified CalledProcessError exceptions
        on failures.
        Note: command is a string and cannot include pipes."""
    try:
        if silent:
            with open(os.devnull, 'w')  as FNULL:
                return subprocess.check_call(command_to_array(command), stdout=FNULL)
        else:
            # Using the defaults, shell=False, no i/o redirection.
            return check_call(command_to_array(command))
    except CalledProcessError as e:
        # We are modifying the error itself for 2 reasons.  1) it WILL contain
        # login credentials when run_mongodump is run, 2) CalledProcessError is
        # slightly not-to-spec (the message variable is blank), which means
        # cronutils.ErrorHandler would report unlabeled stack traces.
        e.message = "%s failed with error code %s" % (e.cmd[0], e.returncode)
        e.cmd = e.cmd[0] + " [arguments stripped for security]"
        raise e