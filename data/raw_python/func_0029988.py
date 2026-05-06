def schema(self, sources=None, tables=None, clean=False, force=False, use_pipeline=False):
        """
        Generate destination schemas.

        :param sources: If specified, build only destination tables for these sources
        :param tables: If specified, build only these tables
        :param clean: Delete tables and partitions first
        :param force: Population tables even if the table isn't empty
        :param use_pipeline: If True, use the build pipeline to determine columns. If False,

        :return: True on success.
        """
        from itertools import groupby
        from operator import attrgetter
        from ambry.etl import Collect, Head
        from ambry.orm.exc import NotFoundError

        self.dstate = self.STATES.BUILDING
        self.commit()  # Workaround for https://github.com/CivicKnowledge/ambry/issues/171

        self.log('---- Schema ----')

        resolved_sources = self._resolve_sources(sources, tables, predicate=lambda s: s.is_processable)

        if clean:
            self.dataset.delete_tables_partitions()
            self.commit()

        # Group the sources by the destination table name
        keyfunc = attrgetter('dest_table')
        for t, table_sources in groupby(sorted(resolved_sources, key=keyfunc), keyfunc):

            if use_pipeline:
                for source in table_sources:
                    pl = self.pipeline(source)

                    pl.cast = [ambry.etl.CastSourceColumns]
                    pl.select_partition = []
                    pl.write = [Head, Collect]
                    pl.final = []

                    self.log_pipeline(pl)

                    pl.run()
                    pl.phase = 'build_schema'
                    self.log_pipeline(pl)

                    for h, c in zip(pl.write[Collect].headers, pl.write[Collect].rows[1]):
                        c = t.add_column(name=h, datatype=type(c).__name__ if c is not None else 'str',
                                         update_existing=True)

                self.log("Populated destination table '{}' from pipeline '{}'"
                         .format(t.name, pl.name))

            else:
                # Get all of the header names, for each source, associating the header position in the table
                # with the header, then sort on the postition. This will produce a stream of header names
                # that may have duplicates, but which is generally in the order the headers appear in the
                # sources. The duplicates are properly handled when we add the columns in add_column()

                self.commit()

                def source_cols(source):
                    if source.is_partition and not source.source_table_exists:
                        return enumerate(source.partition.table.columns)

                    else:
                        return enumerate(source.source_table.columns)

                columns = sorted(set([(i, col.dest_header, col.datatype, col.description, col.has_codes)
                                      for source in table_sources for i, col in source_cols(source)]))

                initial_count = len(t.columns)

                for pos, name, datatype, desc, has_codes in columns:

                    kwds = dict(
                        name=name,
                        datatype=datatype,
                        description=desc,
                        update_existing=True
                    )


                    try:
                        extant = t.column(name)
                    except NotFoundError:
                        extant = None

                    if extant is None or not extant.description:
                        kwds['description'] = desc

                    c = t.add_column(**kwds)


                final_count = len(t.columns)

                if final_count > initial_count:
                    diff = final_count - initial_count

                    self.log("Populated destination table '{}' from source table '{}' with {} columns"
                             .format(t.name, source.source_table.name, diff))

        self.commit()

        return True