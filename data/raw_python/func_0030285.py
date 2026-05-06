def translate_path(self, pth):
        """Translate a /-separated PATH to the local filename syntax."""
        # initially copied from SimpleHTTPServer
        words = pth.split('/')
        words = filter(None, words)
        pth = self.location
        for word in words:
            # Do not allow path separators other than /,
            # drive names and . ..
            drive, word = os.path.splitdrive(word)
            head, word = os.path.split(word)
            if drive or head or word in (os.curdir, os.pardir):
                return None
            pth = os.path.join(pth, word)

        assert pth.startswith(self.location + '/')
        assert pth == path.normpath(pth)
        return pth