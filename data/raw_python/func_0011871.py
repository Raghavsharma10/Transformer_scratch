def _homogenize_linesep(line):
    """Enforce line separators to be the right one depending on platform."""
    token = str(uuid.uuid4())
    line = line.replace(os.linesep, token).replace("\n", "").replace("\r", "")
    return line.replace(token, os.linesep)