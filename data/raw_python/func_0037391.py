def _robust_remove(path):
        """
        Remove the directory specified by `path`. Because we can't determine
        directly if the path is in use, and on Windows, it's not possible to
        remove a path if it is in use, retry a few times until the call
        succeeds.
        """
        tries = itertools.count()
        max_tries = 50
        while os.path.isdir(path):
            try:
                shutil.rmtree(path)
            except WindowsError:
                if next(tries) >= max_tries:
                    raise
                time.sleep(0.2)