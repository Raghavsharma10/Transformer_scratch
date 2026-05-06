def from_git(self, path=None, prefer_daily=False):
        """Use Git to determine the package version.

           This routine uses the __file__ value of the caller to determine
           which Git repository root to use.
        """
        if self._version is None:
            frame = caller(1)
            path = frame.f_globals.get('__file__') or '.'
            providers = ([git_day, git_version] if prefer_daily
                         else [git_version, git_day])
            for provider in providers:
                if self._version is not None:
                    break
                try:
                    with cd(path):
                        self._version = provider()
                except CalledProcessError:
                    pass
                except OSError as e:
                    if e.errno != errno.ENOENT:
                        raise
        return self