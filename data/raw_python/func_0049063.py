def set_locale(self, language_type, script_type):
        """Specifies a language and script type for ``DisplayText`` fields in this form.

        Setting a locale to something other than the default locale may
        affect the ``Metadata`` in this form.

        If multiple locales are available for managing translations, the
        ``Metadata`` indicates the fields are unset as they may be
        returning a defeult value based on the default locale.

        arg:    language_type (osid.type.Type): the language type
        arg:    script_type (osid.type.Type): the script type
        raise:  NullArgument - ``language_type`` or ``script_type`` is
                null
        raise:  Unsupported - ``language_type`` and ``script_type`` not
                available from ``get_locales()``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Someday I might have a real implementation
        # For now only default Locale is supported
        self._locale_map['languageType'] = language_type
        self._locale_map['scriptType'] = script_type