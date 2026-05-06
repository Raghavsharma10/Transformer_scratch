def _ingest_source(self, source, ps, force=None):
        """Ingest a single source"""
        from ambry.bundle.process import call_interval

        try:

            from ambry.orm.exc import NotFoundError

            if not source.is_partition and source.datafile.exists:

                if not source.datafile.is_finalized:
                    source.datafile.remove()
                elif force:
                    source.datafile.remove()
                else:
                    ps.update(
                        message='Source {} already ingested, skipping'.format(source.name),
                        state='skipped')
                    return True

            if source.is_partition:
                # Check if the partition exists
                try:
                    self.library.partition(source.ref)
                except NotFoundError:
                    # Maybe it is an internal reference, in which case we can just delay
                    # until the partition is built
                    ps.update(message="Not Ingesting {}: referenced partition '{}' does not exist"
                              .format(source.name, source.ref), state='skipped')
                    return True

            source.state = source.STATES.INGESTING

            iterable_source, source_pipe = self.source_pipe(source, ps)

            if not source.is_ingestible:
                ps.update(message='Not an ingestiable source: {}'.format(source.name),
                          state='skipped', source=source)
                source.state = source.STATES.NOTINGESTABLE

                return True

            ps.update('Ingesting {} from {}'.format(source.spec.name, source.url or source.generator),
                      item_type='rows', item_count=0)

            @call_interval(5)
            def ingest_progress_f(i):
                (desc, n_records, total, rate) = source.datafile.report_progress()

                ps.update(
                    message='Ingesting {}: rate: {}'.format(source.spec.name, rate), item_count=n_records)

            source.datafile.load_rows(iterable_source,
                                      callback=ingest_progress_f,
                                      limit=500 if self.limited_run else None,
                                      intuit_type=True, run_stats=False)

            if source.datafile.meta['warnings']:
                for w in source.datafile.meta['warnings']:
                    self.error("Ingestion error: {}".format(w))

            ps.update(message='Ingested to {}'.format(source.datafile.syspath))

            ps.update(message='Updating tables and specs for {}'.format(source.name))

            # source.update_table()  # Generate the source tables.
            source.update_spec()  # Update header_lines, start_line, etc.

            if self.limited_run:
                source.end_line = None  # Otherwize, it will be 500

            self.build_source_files.sources.objects_to_record()

            ps.update(message='Ingested {}'.format(source.datafile.path), state='done')
            source.state = source.STATES.INGESTED
            self.commit()

            return True

        except Exception as e:
            import traceback
            from ambry.util import qualified_class_name

            ps.update(
                message='Source {} failed with exception: {}'.format(source.name, e),
                exception_class=qualified_class_name(e),
                exception_trace=str(traceback.format_exc()),
                state='error'
            )

            source.state = source.STATES.INGESTING + '_error'
            self.commit()
            return False