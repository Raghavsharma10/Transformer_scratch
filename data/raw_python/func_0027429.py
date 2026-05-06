def rebase_opt(self):
        """Determine which option name to use."""
        if not hasattr(self, '_rebase_opt'):
            # out = b"MAJOR.MINOR.REVISION" // b"3.4.19" or b"4.0.0"
            out, err = Popen(
                ['cleancss', '--version'], stdout=PIPE).communicate()
            ver = int(out[:out.index(b'.')])
            self._rebase_opt = ['--root', self.root] if ver == 3 else []
        return self._rebase_opt