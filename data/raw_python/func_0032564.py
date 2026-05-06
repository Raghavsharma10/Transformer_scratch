def _replace(self, url):
        """
        Change URLs with absolute paths so they are rooted at the correct
        location.
        """
        segments = url.split('/')
        if segments[0] == '':
            root = self.rootURL(self.request)
            if segments[1] == 'Mantissa':
                root = root.child('static').child('mantissa-base')
                segments = segments[2:]
            elif segments[1] in self.installedOfferingNames:
                root = root.child('static').child(segments[1])
                segments = segments[2:]
            for seg in segments:
                root = root.child(seg)
            return str(root)
        return url