def cygpath(filename):
    """Convert a cygwin path into a windows style path"""
    if sys.platform == 'cygwin':
        proc = Popen(['cygpath', '-am', filename], stdout=PIPE)
        return proc.communicate()[0].strip()
    else:
        return filename