def ffprobe(input_file, verbose=False):
    """Runs ffprobe on file and returns python dict with result"""
    if isinstance(input_file, FileObject):
        exists = input_file.exists
        path = input_file.path
    elif type(input_file) in string_types:
        exists = os.path.exists(input_file)
        path = input_file
    else:
        raise TypeError("input_path must be of string or FileObject type")
    if not exists:
        logging.error("ffprobe: file does not exist ({})".format(input_file))
        return False
    cmd = [
            "ffprobe",
            "-show_format",
            "-show_streams",
            "-print_format", "json",
            path
        ]
    FNULL = open(os.devnull, "w")
    if verbose:
        logging.debug("Executing {}".format(" ".join(cmd)))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    res = decode_if_py3(proc.stdout.read())
    proc.wait()
    if proc.returncode:
        if verbose:
            logging.error("Unable to read media file {}\n\n{}\n\n".format(input_file, indent(proc.stderr.read())))
        else:
            logging.warning("Unable to read media file {}".format(input_file))
        return False
    return json.loads(res)