def command_line(cmd):
    """Handle the command line call

    keyword arguments:
    cmd = a list

    return
    0 if error
    or a string for the command line output
    """
    try:
        s = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        s = s.stdout.read()

        return s.strip()

    except subprocess.CalledProcessError:
        return 0