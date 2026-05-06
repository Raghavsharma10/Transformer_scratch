def build_source(self, stage, source, ps, force=False):
        """Build a single source"""
        from ambry.bundle.process import call_interval

        assert source.is_processable, source.name

        if source.state == self.STATES.BUILT and not force:
            ps.update(message='Source {} already built'.format(source.name), state='skipped')
            return

        pl = self.pipeline(source, ps=ps)

        source.state = self.STATES.BUILDING

        # Doing this before hand to get at least some information about the pipline,
        # in case there is an error during the run. It will get overwritten with more information
        # after successful run
        self.log_pipeline(pl)

        try:

            source_name = source.name  # In case the source drops out of the session, which is does.
            s_vid = source.vid

            ps.update(message='Running pipeline {}'.format(pl.name), s_vid=s_vid, item_type='rows', item_count=0)

            @call_interval(5)
            def run_progress_f(sink_pipe, rows):
                (n_records, rate) = sink_pipe.report_progress()
                if n_records > 0:
                    ps.update(message='Running pipeline {}: rate: {}'
                              .format(pl.name, rate),
                              s_vid=s_vid, item_type='rows', item_count=n_records)


            pl.run(callback=run_progress_f)

            # Run the final routines at the end of the pipelin
            for f in pl.final:
                ps.update(message='Run final routine: {}'.format(f.__name__))
                f(pl)

            ps.update(message='Finished building source')

        except:
            self.log_pipeline(pl)
            raise

        self.commit()

        try:
            partitions = list(pl[ambry.etl.PartitionWriter].partitions)
            ps.update(message='Finalizing segment partition',
                      item_type='partitions', item_total=len(partitions), item_count=0)
            for i, p in enumerate(partitions):

                ps.update(message='Finalizing segment partition {}'.format(p.name), item_count=i, p_vid=p.vid)

                try:
                    p.finalize()
                except AttributeError:
                    print(self.table(p.table_name))
                    raise

                # FIXME Shouldn't need to do this commit, but without it, some stats get added multiple
                # times, causing an error later. Probably could be avoided by adding the stats to the
                # collection in the dataset

                self.commit()

        except IndexError:
            self.error("Pipeline didn't have a PartitionWriters, won't try to finalize")

        self.log_pipeline(pl)
        source.state = self.STATES.BUILT

        self.commit()

        return source.name