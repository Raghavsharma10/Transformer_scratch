def _get_translation(self, lang):
        """Add a new translation language to the live gettext translator"""

        try:
            return self._translations[lang]
        except KeyError:
            # The fact that `fallback=True` is not the default is a serious design flaw.
            rv = self._translations[lang] = gettext.translation(self._domain, localedir=localedir, languages=[lang],
                                                                fallback=True)
            return rv