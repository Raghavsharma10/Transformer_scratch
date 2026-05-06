def is_installed(gemname, version=None):
    """Check if a gem is installed."""
    cmdline = ['gem', 'list', '-i', gemname]
    if version:
        cmdline.extend(['-v', version])
    try:
        subprocess.check_output(cmdline, shell=False)
        return True
    except (OSError, subprocess.CalledProcessError) as err:
        if err.returncode == 1:
            return False
        else:
            raise error.ButcherError(
                'Failure running gem. Error was: %s. Output: %s', err,
                err.output)