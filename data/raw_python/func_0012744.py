def parse_error(output_dir):
    """Add contents of stderr and eplusout.err and put it in the exception message.

    :param output_dir: str
    :return: str
    """
    sys.stderr.seek(0)
    std_err = sys.stderr.read().decode('utf-8')
    err_file = os.path.join(output_dir, "eplusout.err")
    if os.path.isfile(err_file):
        with open(err_file, "r") as f:
            ep_err = f.read()
    else:
        ep_err = "<File not found>"
    message = "\r\n{std_err}\r\nContents of EnergyPlus error file at {err_file}\r\n{ep_err}".format(**locals())
    return message