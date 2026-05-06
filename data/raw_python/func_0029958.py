def pipeline(self, source=None, phase='build', ps=None):
        """
        Construct the ETL pipeline for all phases. Segments that are not used for the current phase
        are filtered out later.

        :param source: A source object, or a source string name
        :return: an etl Pipeline
        """
        from ambry.etl.pipeline import Pipeline, PartitionWriter
        from ambry.dbexceptions import ConfigurationError

        if source:
            source = self.source(source) if isinstance(source, string_types) else source
        else:
            source = None

        sf, sp = self.source_pipe(source, ps) if source else (None, None)


        pl = Pipeline(self, source=sp)

        # Get the default pipeline, from the config at the head of this file.
        try:
            phase_config = self.default_pipelines[phase]
        except KeyError:
            phase_config = None  # Ok for non-conventional pipe names

        if phase_config:
            pl.configure(phase_config)

        # Find the pipe configuration, from the metadata
        pipe_config = None
        pipe_name = None
        if source and source.pipeline:
            pipe_name = source.pipeline
            try:
                pipe_config = self.metadata.pipelines[pipe_name]
            except KeyError:
                raise ConfigurationError("Pipeline '{}' declared in source '{}', but not found in metadata"
                                         .format(source.pipeline, source.name))
        else:
            pipe_name, pipe_config = self._find_pipeline(source, phase)

        if pipe_name:
            pl.name = pipe_name
        else:
            pl.name = phase

        pl.phase = phase

        # The pipe_config can either be a list, in which case it is a list of pipe pipes for the
        # augment segment or it could be a dict, in which case each is a list of pipes
        # for the named segments.

        def apply_config(pl, pipe_config):

            if isinstance(pipe_config, (list, tuple)):
                # Just convert it to dict form for the next section

                # PartitionWriters are always moved to the 'store' section
                store, body = [], []

                for pipe in pipe_config:
                    store.append(pipe) if isinstance(pipe, PartitionWriter) else body.append(pipe)

                pipe_config = dict(body=body, store=store)

            if pipe_config:
                pl.configure(pipe_config)

        apply_config(pl, pipe_config)

        # One more time, for the configuration for 'all' phases
        if 'all' in self.metadata.pipelines:
            apply_config(pl, self.metadata.pipelines['all'])

        # Allows developer to over ride pipe configuration in code

        self.edit_pipeline(pl)

        try:

            pl.dest_table = source.dest_table_name
            pl.source_table = source.source_table.name
            pl.source_name = source.name
        except AttributeError:
            pl.dest_table = None

        return pl