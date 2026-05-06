def build(self, sources=None, tables=None, stage=None, force=False):
        """
        :param phase:
        :param stage:
        :param sources: Source names or destination table names.
        :return:
        """

        from operator import attrgetter
        from itertools import groupby
        from .concurrent import build_mp, unify_mp
        from ambry.bundle.events import TAG

        self.log('==== Building ====')
        self.state = self.STATES.BUILDING
        self.commit()

        class SourceSet(object):
            """Container for sources that can reload them after they get expired from the session"""
            def __init__(self, bundle, v):
                self.bundle = bundle
                self.sources = v
                self._s_vids = [s.vid for s in self.sources]

            def reload(self):
                self.sources = [self.bundle.source(vid) for vid in self._s_vids]

            def __iter__(self):
                for s in self.sources:
                    yield s

            def __len__(self):
                return len(self._s_vids)

        self._run_events(TAG.BEFORE_BUILD, 0)

        resolved_sources = SourceSet(self, self._resolve_sources(sources, tables, stage=stage,
                                                                 predicate=lambda s: s.is_processable))
        with self.progress.start('build', stage, item_total=len(resolved_sources)) as ps:

            if len(resolved_sources) == 0:
                ps.update(message='No sources', state='skipped')
                self.log('No processable sources, skipping build stage {}'.format(stage))
                return True

            if not self.pre_build(force):
                ps.update(message='Pre-build failed', state='skipped')
                return False

            if force:
                self._reset_build(resolved_sources)

            resolved_sources.reload()

            e = [
                (stage, SourceSet(self, list(stage_sources)))
                for stage, stage_sources in groupby(sorted(resolved_sources, key=attrgetter('stage')),
                                                    attrgetter('stage'))

                ]

            for stage, stage_sources in e:

                stage_sources.reload()

                for s in stage_sources:
                    s.state = self.STATES.WAITING
                self.commit()

                stage_sources.reload()

                self.log('Processing {} sources, stage {} ; first 10: {}'
                         .format(len(stage_sources), stage, [x.name for x in stage_sources.sources[:10]]))
                self._run_events(TAG.BEFORE_BUILD, stage)

                if self.multi:

                    try:
                        # The '1' for chunksize ensures that the subprocess only gets one
                        # source to build. Combined with maxchildspertask = 1 in the pool,
                        # each process will only handle one source before exiting.

                        args = [(self.identity.vid, stage, source.vid, force) for source in stage_sources]
                        pool = self.library.process_pool(limited_run=self.limited_run)
                        r = pool.map_async(build_mp, args, 1)
                        completed_sources = r.get()

                        ps.add('Finished MP building {} sources. Starting MP coalescing'
                               .format(len(completed_sources)))

                        partition_names = [(self.identity.vid, k) for k, v
                                           in self.collect_segment_partitions().items()]

                        r = pool.map_async(unify_mp, partition_names, 1)

                        completed_partitions = r.get()

                        ps.add('Finished MP coalescing {} partitions'.format(len(completed_partitions)))

                        pool.close()
                        pool.join()

                    except KeyboardInterrupt:
                        self.log('Got keyboard interrrupt; terminating workers')
                        pool.terminate()

                else:

                    for i, source in enumerate(stage_sources):
                        id_ = ps.add(message='Running source {}'.format(source.name),
                                     source=source, item_count=i, state='running')

                        self.build_source(stage, source, ps, force=force)

                        ps.update(message='Finished processing source', state='done')

                        # This bit seems to solve a problem where the records from the ps.add above
                        # never gets closed out.
                        ps.get(id_).state = 'done'
                        self.progress.commit()

                    self.unify_partitions()

                self._run_events(TAG.AFTER_BUILD, stage)

        self.state = self.STATES.BUILT
        self.commit()

        self._run_events(TAG.AFTER_BUILD, 0)

        self.close_session()

        self.log('==== Done Building ====')
        self.buildstate.commit()
        self.state = self.STATES.BUILT
        self.commit()
        return True