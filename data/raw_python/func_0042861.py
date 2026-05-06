def get_tuids_from_files(
            self,
            files,
            revision,
            going_forward=False,
            repo=None,
            use_thread=True,
            max_csets_proc=30
        ):
        """
        Gets the TUIDs for a set of files, at a given revision.
        list(tuids) is an array of tuids, one tuid for each line, in order, and `null` if no tuid assigned

        Uses frontier updating to build and maintain the tuids for
        the given set of files. Use changelog to determine what revisions
        to process and get the files that need to be updated by looking
        at the diffs. If the latestFileMod table is empty, for any file,
        we perform an annotation-based update.

        This function assumes the newest file names are given, if they
        are not, then no TUIDs are returned for that file.

        IMPORTANT:
        If repo is set to None, the service will check if the revision is in
        the correct branch (to prevent catastrophic failures down the line) - this
        results in one extra changeset log call per request.
        If repo is set to something other than None, then we assume that the caller has already
        checked this and is giving a proper branch for the revision.

        :param files: list of files
        :param revision: revision to get files at
        :param repo: Branch to get files from (mozilla-central, or try)
        :param disable_thread: Disables the thread that spawns if the number of files to process exceeds the
                               threshold set by FILES_TO_PROCESS_THRESH.
        :param going_forward: When set to true, the frontiers always get updated to the given revision
                              even if we can't find a file's frontier. Otherwise, if a frontier is too far,
                              the latest revision will not be updated.
        :return: The following tuple which contains:
                    ([list of (file, list(tuids)) tuples], True/False if completed or not)
        """
        completed = True

        if repo is None:
            repo = self.config.hg.branch
            check = self._check_branch(revision, repo)
            if not check:
                # Error was already output by _check_branch
                return [(file, []) for file in files], completed

        if repo in ('try',):
            # We don't need to keep latest file revisions
            # and other related things for this condition.

            # Enable the 'try' repo calls with ENABLE_TRY
            if ENABLE_TRY:
                return self._get_tuids_from_files_try_branch(files, revision), completed
            return [(file, []) for file in files], completed

        result = []
        revision = revision[:12]
        files = [file.lstrip('/') for file in files]
        frontier_update_list = []

        total = len(files)
        latestFileMod_inserts = {}
        new_files = []

        log_existing_files = []
        for count, file in enumerate(files):
            # Go through all requested files and
            # either update their frontier or add
            # them to the DB through an initial annotation.

            if DEBUG:
                Log.note(" {{percent|percent(decimal=0)}}|{{file}}", file=file, percent=count / total)

            with self.conn.transaction() as t:
                latest_rev = self._get_latest_revision(file, transaction=t)
                already_ann = self._get_annotation(revision, file, transaction=t)

            # Check if the file has already been collected at
            # this revision and get the result if so
            if already_ann:
                result.append((file,self.destringify_tuids(already_ann)))
                if going_forward:
                    latestFileMod_inserts[file] = (file, revision)
                log_existing_files.append('exists|' + file)
                continue
            elif already_ann == '':
                result.append((file,[]))
                if going_forward:
                    latestFileMod_inserts[file] = (file, revision)
                log_existing_files.append('removed|' + file)
                continue

            if (latest_rev and latest_rev[0] != revision):
                # File has a frontier, let's update it
                if DEBUG:
                    Log.note("Will update frontier for file {{file}}.", file=file)
                frontier_update_list.append((file, latest_rev[0]))
            elif latest_rev == revision:
                with self.conn.transaction() as t:
                    t.execute("DELETE FROM latestFileMod WHERE file = " + quote_value(file))
                new_files.append(file)
                Log.note(
                    "Missing annotation for existing frontier - readding: "
                    "{{rev}}|{{file}} ",
                    file=file, rev=revision
                )
            else:
                Log.note(
                    "Frontier update - adding: "
                    "{{rev}}|{{file}} ",
                    file=file, rev=revision
                )
                new_files.append(file)

        if DEBUG:
            Log.note(
                "Frontier update - already exist in DB: "
                "{{rev}} || {{file_list}} ",
                file_list=str(log_existing_files), rev=revision
            )
        else:
            Log.note(
                "Frontier update - already exist in DB for {{rev}}: "
                    "{{count}}/{{total}} | {{percent|percent}}",
                count=str(len(log_existing_files)), total=str(len(files)),
                rev=revision, percent=len(log_existing_files)/len(files)
            )

        if len(latestFileMod_inserts) > 0:
            with self.conn.transaction() as transaction:
                for _, inserts_list in jx.groupby(latestFileMod_inserts.values(), size=SQL_BATCH_SIZE):
                    transaction.execute(
                        "INSERT OR REPLACE INTO latestFileMod (file, revision) VALUES " +
                        sql_list(quote_list(i) for i in inserts_list)
                    )

        def update_tuids_in_thread(
                new_files,
                frontier_update_list,
                revision,
                using_thread,
                please_stop=None
            ):
            try:
                # Processes the new files and files which need their frontier updated
                # outside of the main thread as this can take a long time.
                result = []

                latestFileMod_inserts = {}
                if len(new_files) > 0:
                    # File has never been seen before, get it's initial
                    # annotation to work from in the future.
                    tmp_res = self.get_tuids(new_files, revision, commit=False)
                    if tmp_res:
                        result.extend(tmp_res)
                    else:
                        Log.note("Error occured for files " + str(new_files) + " in revision " + revision)

                    # If this file has not been seen before,
                    # add it to the latest modifications, else
                    # it's already in there so update its past
                    # revisions.
                    for file in new_files:
                        latestFileMod_inserts[file] = (file, revision)

                Log.note("Finished updating frontiers. Updating DB table `latestFileMod`...")
                if len(latestFileMod_inserts) > 0:
                    with self.conn.transaction() as transaction:
                        for _, inserts_list in jx.groupby(latestFileMod_inserts.values(), size=SQL_BATCH_SIZE):
                            transaction.execute(
                                "INSERT OR REPLACE INTO latestFileMod (file, revision) VALUES " +
                                sql_list(quote_list(i) for i in inserts_list)
                            )

                # If we have files that need to have their frontier updated, do that now
                if len(frontier_update_list) > 0:
                    tmp = self._update_file_frontiers(
                        frontier_update_list,
                        revision,
                        going_forward=going_forward,
                        max_csets_proc=max_csets_proc
                    )
                    result.extend(tmp)

                if using_thread:
                    self.pcdaemon.update_totals(0, len(result))
                Log.note("Completed work overflow for revision {{cset}}", cset=revision)
                return result
            except Exception as e:
                Log.warning("Thread dead becasue of problem", cause=e)
                return []

        threaded = False
        if use_thread:
            # If there are too many files to process, start a thread to do
            # that work and return completed as False.
            if (len(new_files) + len(frontier_update_list) > FILES_TO_PROCESS_THRESH):
                threaded = True

        if threaded:
            completed = False
            Log.note("Incomplete response given")
            Thread.run(
                'get_tuids_from_files (' + Random.base64(9) + ")",
                update_tuids_in_thread, new_files, frontier_update_list, revision, threaded
            )
        else:
            result.extend(
                update_tuids_in_thread(new_files, frontier_update_list, revision, threaded)
            )

        self.pcdaemon.update_totals(len(files), len(result))
        return result, completed