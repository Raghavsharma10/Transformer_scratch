def set_vassals_wrapper_params(self, wrapper=None, overrides=None, fallbacks=None):
        """Binary wrapper for vassals parameters.

        :param str|unicode wrapper: Set a binary wrapper for vassals.

        :param str|unicode|list[str|unicode] overrides: Set a binary wrapper for vassals to try before the default one

        :param str|unicode|list[str|unicode] fallbacks: Set a binary wrapper for vassals to try as a last resort.
            Allows you to specify an alternative binary to execute when running a vassal
            and the default binary_path is not found (or returns an error).

        """
        self._set('emperor-wrapper', wrapper)
        self._set('emperor-wrapper-override', overrides, multi=True)
        self._set('emperor-wrapper-fallback', fallbacks, multi=True)

        return self._section