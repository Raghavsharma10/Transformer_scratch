def ingest(self, sources=None, tables=None, stage=None, force=False, load_meta=False):
        """Ingest a set of sources, specified as source objects, source names, or destination tables.
        If no stage is specified, execute the sources in groups by stage.

        Note, however, that when this is called from run_stage, all of the sources have the same stage, so they
        get grouped together. The result it that the stage in the inner loop is the same as the stage being
        run buy run_stage.
        """

        from itertools import groupby
        from ambry.bundle.events import TAG
        from fs.errors import ResourceNotFoundError
        import zlib

        self.log('---- Ingesting ----')

        self.dstate = self.STATES.BUILDING
        self.commit() # WTF? Without this, postgres blocks between table query, and update seq id in source tables.

        key = lambda s: s.stage if s.stage else 1

        def not_final_or_delete(s):
            import zlib

            if force:
                return True

            try:
                return s.is_processable and not s.is_ingested and not s.is_built
            except (IOError, zlib.error):
                s.local_datafile.remove()
                return True

        sources = sorted(self._resolve_sources(sources, tables, stage, predicate=not_final_or_delete),
                         key=key)

        if not sources:
            self.log('No sources left to ingest')
            return

        self.state = self.STATES.INGESTING

        count = 0
        errors = 0

        self._run_events(TAG.BEFORE_INGEST, 0)
        # Clear out all ingested files that are malformed
        for s in self.sources:
            if s.is_downloadable:
                df = s.datafile
                try:
                    info = df.info
                    df.close()
                except (ResourceNotFoundError, zlib.error, IOError):
                    df.remove()

        for stage, g in groupby(sources, key):
            sources = [s for s in g if not_final_or_delete(s)]

            if not len(sources):
                continue

            self._run_events(TAG.BEFORE_INGEST, stage)

            stage_errors = self._ingest_sources(sources, stage, force=force)

            errors += stage_errors

            count += len(sources) - stage_errors

            self._run_events(TAG.AFTER_INGEST, stage)
            self.record_stage_state(self.STATES.INGESTING, stage)

        self.state = self.STATES.INGESTED

        try:
            pass
        finally:
            self._run_events(TAG.AFTER_INGEST, 0)

        self.log('Ingested {} sources'.format(count))

        if load_meta:

            if len(sources) == 1:
                iterable_source, source_pipe = self.source_pipe(sources[0])

                try:
                    meta = iterable_source.meta
                    if meta:
                        self.metadata.about.title = meta['title']
                        self.metadata.about.summary = meta['summary']
                        self.build_source_files.bundle_meta.objects_to_record()

                except AttributeError as e:
                    self.warn("Failed to set metadata: {}".format(e))
                    pass
            else:
                self.warn("Didn't not load meta from source. Must have exactly one soruce, got {}".format(len(sources)))

        self.commit()

        if errors == 0:
            return True
        else:
            return False