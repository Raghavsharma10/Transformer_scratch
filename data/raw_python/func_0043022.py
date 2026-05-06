def get_remote_revision(url, branch):
    """
    GET REVISION OF A REMOTE BRANCH
    """
    proc = Process("git remote revision", ["git", "ls-remote", url, "refs/heads/" + branch])

    try:
        while True:
            raw_line = proc.stdout.pop()
            line = raw_line.strip().decode('utf8')
            if not line:
                continue
            return line.split("\t")[0]
    finally:
        try:
            proc.join()
        except Exception:
            pass