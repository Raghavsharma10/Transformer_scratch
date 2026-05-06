def add_localedir_translations(self, localedir):
        """Merge translations from localedir."""
        global _localedirs
        if localedir in self.localedirs:
            return
        self.localedirs.append(localedir)
        full_localedir = os.path.join(localedir, 'locale')
        if os.path.exists(full_localedir):
            translation = self._new_gnu_trans(full_localedir)
            self.merge(translation)