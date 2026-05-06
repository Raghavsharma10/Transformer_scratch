def _is_always_unsatisfied(self):
        """Returns whether this requirement is always unsatisfied

        This would happen in cases where we can't determine the version
        from the filename.

        """
        # If this is a github sha tarball, then it is always unsatisfied
        # because the url has a commit sha in it and not the version
        # number.
        url = self._url()
        if url:
            filename = filename_from_url(url)
            if filename.endswith(ARCHIVE_EXTENSIONS):
                filename, ext = splitext(filename)
                if is_git_sha(filename):
                    return True
        return False