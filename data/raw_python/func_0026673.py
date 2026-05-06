def load_stubs(self, log_mem=False):
        """Load all events in their `stub` (name, alias, etc only) form.

        Used in `update` mode.
        """
        # Initialize parameter related to diagnostic output of memory usage
        if log_mem:
            import psutil
            process = psutil.Process(os.getpid())
            rss = process.memory_info().rss
            LOG_MEMORY_INT = 1000
            MEMORY_LIMIT = 1000.0

        def _add_stub_manually(_fname):
            """Create and add a 'stub' by manually loading parameters from
            JSON files.

            Previously this was done by creating a full `Entry` instance, then
            using the `Entry.get_stub()` method to trim it down.  This was very
            slow and memory intensive, hence this improved approach.
            """
            # FIX: should this be ``fi.endswith(``.gz')`` ?
            fname = uncompress_gz(_fname) if '.gz' in _fname else _fname

            stub = None
            stub_name = None
            with codecs.open(fname, 'r') as jfil:
                # Load the full JSON file
                data = json.load(jfil, object_pairs_hook=OrderedDict)
                # Extract the top-level keys (should just be the name of the
                # entry)
                stub_name = list(data.keys())
                # Make sure there is only a single top-level entry
                if len(stub_name) != 1:
                    err = "json file '{}' has multiple keys: {}".format(
                        fname, list(stub_name))
                    self._log.error(err)
                    raise ValueError(err)
                stub_name = stub_name[0]

                # Make sure a non-stub entry doesnt already exist with this
                # name
                if stub_name in self.entries and not self.entries[
                        stub_name]._stub:
                    err_str = (
                        "ERROR: non-stub entry already exists with name '{}'"
                        .format(stub_name))
                    self.log.error(err_str)
                    raise RuntimeError(err_str)

                # Remove the outmost dict level
                data = data[stub_name]
                # Create a new `Entry` (subclass) instance
                proto = self.proto
                stub = proto(catalog=self, name=stub_name, stub=True)
                # Add stub parameters if they are available
                if proto._KEYS.ALIAS in data:
                    stub[proto._KEYS.ALIAS] = data[proto._KEYS.ALIAS]
                if proto._KEYS.DISTINCT_FROM in data:
                    stub[proto._KEYS.DISTINCT_FROM] = data[
                        proto._KEYS.DISTINCT_FROM]
                if proto._KEYS.RA in data:
                    stub[proto._KEYS.RA] = data[proto._KEYS.RA]
                if proto._KEYS.DEC in data:
                    stub[proto._KEYS.DEC] = data[proto._KEYS.DEC]
                if proto._KEYS.DISCOVER_DATE in data:
                    stub[proto._KEYS.DISCOVER_DATE] = data[
                        proto._KEYS.DISCOVER_DATE]
                if proto._KEYS.SOURCES in data:
                    stub[proto._KEYS.SOURCES] = data[
                        proto._KEYS.SOURCES]

            # Store the stub
            self.entries[stub_name] = stub
            self.log.debug("Added stub for '{}'".format(stub_name))

        currenttask = 'Loading entry stubs'
        files = self.PATHS.get_repo_output_file_list()
        for ii, _fname in enumerate(pbar(files, currenttask)):
            # Run normally
            # _add_stub(_fname)

            # Run 'manually' (extract stub parameters directly from JSON)
            _add_stub_manually(_fname)

            if log_mem:
                rss = process.memory_info().rss / 1024 / 1024
                if ii % LOG_MEMORY_INT == 0 or rss > MEMORY_LIMIT:
                    log_memory(self.log, "\nLoaded stub {}".format(ii),
                               logging.INFO)
                    if rss > MEMORY_LIMIT:
                        err = (
                            "Memory usage {}, has exceeded {} on file {} '{}'".
                            format(rss, MEMORY_LIMIT, ii, _fname))
                        self.log.error(err)
                        raise RuntimeError(err)

        return self.entries