def get_tuids(self, files, revision, commit=True, chunk=50, repo=None):
        '''
        Wrapper for `_get_tuids` to limit the number of annotation calls to hg
        and separate the calls from DB transactions. Also used to simplify `_get_tuids`.

        :param files:
        :param revision:
        :param commit:
        :param chunk:
        :param repo:
        :return:
        '''
        results = []
        revision = revision[:12]

        # For a single file, there is no need
        # to put it in an array when given.
        if not isinstance(files, list):
            files = [files]
        if repo is None:
            repo = self.config.hg.branch

        for _, new_files in jx.groupby(files, size=chunk):
            for count, file in enumerate(new_files):
                new_files[count] = file.lstrip('/')

            annotations_to_get = []
            for file in new_files:
                with self.conn.transaction() as t:
                    already_ann = self._get_annotation(revision, file, transaction=t)
                if already_ann:
                    results.append((file, self.destringify_tuids(already_ann)))
                elif already_ann == '':
                    results.append((file, []))
                else:
                    annotations_to_get.append(file)

            if not annotations_to_get:
                # No new annotations to get, so get next set
                continue

            # Get all the annotations in parallel and
            # store in annotated_files
            annotated_files = [None] * len(annotations_to_get)
            threads = [
                Thread.run(
                    str(thread_count),
                    self._get_hg_annotate,
                    revision,
                    annotations_to_get[thread_count],
                    annotated_files,
                    thread_count,
                    repo
                )
                for thread_count, _ in enumerate(annotations_to_get)
            ]
            for t in threads:
                t.join()

            # Help for memory, because `chunk` (or a lot of)
            # threads are started at once.
            del threads

            with self.conn.transaction() as transaction:
                results.extend(
                    self._get_tuids(
                        transaction, annotations_to_get, revision, annotated_files, commit=commit, repo=repo
                    )
                )

        # Help for memory
        gc.collect()
        return results