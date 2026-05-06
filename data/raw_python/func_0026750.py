def sanitize(self):
        """Sanitize the data (sort it, etc.) before writing it to disk.

        Template method that can be overridden in each catalog's subclassed
        `Entry` object.
        """
        name = self[self._KEYS.NAME]

        aliases = self.get_aliases(includename=False)
        if name not in aliases:
            # Assign the first source to alias, if not available assign us.
            if self._KEYS.SOURCES in self:
                self.add_quantity(self._KEYS.ALIAS, name, '1')
                if self._KEYS.ALIAS not in self:
                    source = self.add_self_source()
                    self.add_quantity(self._KEYS.ALIAS, name, source)
            else:
                source = self.add_self_source()
                self.add_quantity(self._KEYS.ALIAS, name, source)

        if self._KEYS.ALIAS in self:
            self[self._KEYS.ALIAS].sort(
                key=lambda key: alias_priority(name, key[QUANTITY.VALUE]))
        else:
            self._log.error(
                'There should be at least one alias for `{}`.'.format(name))

        if self._KEYS.PHOTOMETRY in self:
            self[self._KEYS.PHOTOMETRY].sort(
                key=lambda x: ((float(x[PHOTOMETRY.TIME]) if
                                isinstance(x[PHOTOMETRY.TIME],
                                           (basestring, float, int))
                                else min([float(y) for y in
                                          x[PHOTOMETRY.TIME]])) if
                               PHOTOMETRY.TIME in x else 0.0,
                               x[PHOTOMETRY.BAND] if PHOTOMETRY.BAND in
                               x else '',
                               float(x[PHOTOMETRY.MAGNITUDE]) if
                               PHOTOMETRY.MAGNITUDE in x else ''))

        if (self._KEYS.SPECTRA in self and list(
                filter(None, [
                    SPECTRUM.TIME in x for x in self[self._KEYS.SPECTRA]
                ]))):
            self[self._KEYS.SPECTRA].sort(
                key=lambda x: (float(x[SPECTRUM.TIME]) if
                               SPECTRUM.TIME in x else 0.0,
                               x[SPECTRUM.FILENAME] if
                               SPECTRUM.FILENAME in x else '')
            )

        if self._KEYS.SOURCES in self:
            # Remove orphan sources
            source_aliases = [
                x[SOURCE.ALIAS] for x in self[self._KEYS.SOURCES]
            ]
            # Sources with the `PRIVATE` attribute are always retained
            source_list = [
                x[SOURCE.ALIAS] for x in self[self._KEYS.SOURCES]
                if SOURCE.PRIVATE in x
            ]
            for key in self:
                # if self._KEYS.get_key_by_name(key).no_source:
                if (key in [
                        self._KEYS.NAME, self._KEYS.SCHEMA, self._KEYS.SOURCES,
                        self._KEYS.ERRORS
                ]):
                    continue
                for item in self[key]:
                    source_list += item[item._KEYS.SOURCE].split(',')
            new_src_list = sorted(
                list(set(source_aliases).intersection(source_list)))
            new_sources = []
            for source in self[self._KEYS.SOURCES]:
                if source[SOURCE.ALIAS] in new_src_list:
                    new_sources.append(source)
                else:
                    self._log.info('Removing orphaned source from `{}`.'
                                   .format(name))

            if not new_sources:
                del self[self._KEYS.SOURCES]

            self[self._KEYS.SOURCES] = new_sources