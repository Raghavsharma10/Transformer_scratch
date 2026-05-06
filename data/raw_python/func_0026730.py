def _convert_odict_to_classes(self,
                                  data,
                                  clean=False,
                                  merge=True,
                                  pop_schema=True,
                                  compare_to_existing=True,
                                  filter_on={}):
        """Convert `OrderedDict` into `Entry` or its derivative classes."""
        self._log.debug("_convert_odict_to_classes(): {}".format(self.name()))
        self._log.debug("This should be a temporary fix.  Dont be lazy.")

        # Setup filters. Currently only used for photometry.
        fkeys = list(filter_on.keys())

        # Handle 'name'
        name_key = self._KEYS.NAME
        if name_key in data:
            self[name_key] = data.pop(name_key)

        # Handle 'schema'
        schema_key = self._KEYS.SCHEMA
        if schema_key in data:
            # Schema should be re-added every execution (done elsewhere) so
            # just delete the old entry
            if pop_schema:
                data.pop(schema_key)
            else:
                self[schema_key] = data.pop(schema_key)

        # Cleanup 'internal' repository stuff
        if clean:
            # Add data to `self` in ways accomodating 'internal' formats and
            # leeway.  Removes each added entry from `data` so the remaining
            # stuff can be handled normally
            data = self.clean_internal(data)

        # Handle 'sources'
        # ----------------
        src_key = self._KEYS.SOURCES
        if src_key in data:
            # Remove from `data`
            sources = data.pop(src_key)
            self._log.debug("Found {} '{}' entries".format(
                len(sources), src_key))
            self._log.debug("{}: {}".format(src_key, sources))

            for src in sources:
                self.add_source(allow_alias=True, **src)

        # Handle `photometry`
        # -------------------
        photo_key = self._KEYS.PHOTOMETRY
        if photo_key in data:
            photoms = data.pop(photo_key)
            self._log.debug("Found {} '{}' entries".format(
                len(photoms), photo_key))
            phcount = 0
            for photo in photoms:
                skip = False
                for fkey in fkeys:
                    if fkey in photo and photo[fkey] not in filter_on[fkey]:
                        skip = True
                if skip:
                    continue
                self._add_cat_dict(
                    Photometry,
                    self._KEYS.PHOTOMETRY,
                    compare_to_existing=compare_to_existing,
                    **photo)
                phcount += 1
            self._log.debug("Added {} '{}' entries".format(
                phcount, photo_key))

        # Handle `spectra`
        # ---------------
        spec_key = self._KEYS.SPECTRA
        if spec_key in data:
            # When we are cleaning internal data, we don't always want to
            # require all of the normal spectrum data elements.
            spectra = data.pop(spec_key)
            self._log.debug("Found {} '{}' entries".format(
                len(spectra), spec_key))
            for spec in spectra:
                self._add_cat_dict(
                    Spectrum,
                    self._KEYS.SPECTRA,
                    compare_to_existing=compare_to_existing,
                    **spec)

        # Handle `error`
        # --------------
        err_key = self._KEYS.ERRORS
        if err_key in data:
            errors = data.pop(err_key)
            self._log.debug("Found {} '{}' entries".format(
                len(errors), err_key))
            for err in errors:
                self._add_cat_dict(Error, self._KEYS.ERRORS, **err)

        # Handle `models`
        # ---------------
        model_key = self._KEYS.MODELS
        if model_key in data:
            # When we are cleaning internal data, we don't always want to
            # require all of the normal spectrum data elements.
            model = data.pop(model_key)
            self._log.debug("Found {} '{}' entries".format(
                len(model), model_key))
            for mod in model:
                self._add_cat_dict(
                    Model,
                    self._KEYS.MODELS,
                    compare_to_existing=compare_to_existing,
                    **mod)

        # Handle everything else --- should be `Quantity`s
        # ------------------------------------------------
        if len(data):
            self._log.debug("{} remaining entries, assuming `Quantity`".format(
                len(data)))
            # Iterate over remaining keys
            for key in list(data.keys()):
                vals = data.pop(key)
                # All quantities should be in lists of that quantity
                #    E.g. `aliases` is a list of alias quantities
                if not isinstance(vals, list):
                    vals = [vals]
                self._log.debug("{}: {}".format(key, vals))
                for vv in vals:
                    self._add_cat_dict(
                        Quantity,
                        key,
                        check_for_dupes=merge,
                        compare_to_existing=compare_to_existing,
                        **vv)

        if merge and self.dupe_of:
            self.merge_dupes()

        return