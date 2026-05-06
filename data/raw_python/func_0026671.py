def copy_entry_to_entry(self,
                            fromentry,
                            destentry,
                            check_for_dupes=True,
                            compare_to_existing=True):
        """Used by `merge_duplicates`
        """
        self.log.info("Copy entry object '{}' to '{}'".format(fromentry[
            fromentry._KEYS.NAME], destentry[destentry._KEYS.NAME]))

        newsourcealiases = {}
        if self.proto._KEYS.SOURCES in fromentry:
            for source in fromentry[self.proto._KEYS.SOURCES]:
                alias = source.pop(SOURCE.ALIAS)
                newsourcealiases[alias] = source

        newmodelaliases = {}
        if self.proto._KEYS.MODELS in fromentry:
            for model in fromentry[self.proto._KEYS.MODELS]:
                alias = model.pop(MODEL.ALIAS)
                newmodelaliases[alias] = model

        if self.proto._KEYS.ERRORS in fromentry:
            for err in fromentry[self.proto._KEYS.ERRORS]:
                destentry.setdefault(self.proto._KEYS.ERRORS, []).append(err)

        for rkey in fromentry:
            key = fromentry._KEYS.get_key_by_name(rkey)
            if key.no_source:
                continue
            for item in fromentry[key]:
                # isd = False
                if 'source' not in item:
                    raise ValueError("Item has no source!")

                nsid = []
                for sid in item['source'].split(','):
                    if sid in newsourcealiases:
                        source = newsourcealiases[sid]
                        nsid.append(destentry.add_source(**source))
                    else:
                        raise ValueError("Couldn't find source alias!")
                item['source'] = uniq_cdl(nsid)

                if 'model' in item:
                    nmid = []
                    for mid in item['model'].split(','):
                        if mid in newmodelaliases:
                            model = newmodelaliases[mid]
                            nmid.append(destentry.add_model(**model))
                        else:
                            raise ValueError("Couldn't find model alias!")
                    item['model'] = uniq_cdl(nmid)

                if key == ENTRY.PHOTOMETRY:
                    destentry.add_photometry(
                        compare_to_existing=compare_to_existing,
                        **item)
                elif key == ENTRY.SPECTRA:
                    destentry.add_spectrum(
                        compare_to_existing=compare_to_existing,
                        **item)
                elif key == ENTRY.ERRORS:
                    destentry.add_error(**item)
                elif key == ENTRY.MODELS:
                    continue
                else:
                    destentry.add_quantity(
                        compare_to_existing=compare_to_existing,
                        check_for_dupes=False, quantities=key, **item)

        return