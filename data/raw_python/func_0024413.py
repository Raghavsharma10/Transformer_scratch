def get_staged_files():
    """Get all files staged for the current commit.
    """
    proc = subprocess.Popen(('git', 'status', '--porcelain'),
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    out, _ = proc.communicate()
    staged_files = modified_re.findall(out)
    return staged_files