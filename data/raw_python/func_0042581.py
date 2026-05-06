def csetLog_maintenance(self, please_stop=None):
        '''
        Handles deleting old csetLog entries and timestamping
        revisions once they pass the length for permanent
        storage for deletion later.
        :param please_stop:
        :return:
        '''
        while not please_stop:
            try:
                # Wait until something signals the maintenance cycle
                # to begin (or end).
                (self.maintenance_signal | please_stop).wait()

                if please_stop:
                    break
                if self.disable_maintenance:
                    continue

                # Reset signal so we don't request
                # maintenance infinitely.
                with self.maintenance_signal.lock:
                    self.maintenance_signal._go = False

                with self.working_locker:
                    all_data = None
                    with self.conn.transaction() as t:
                        all_data = sorted(
                            t.get("SELECT revnum, revision, timestamp FROM csetLog"),
                            key=lambda x: int(x[0])
                        )

                    # Restore maximum permanents (if overflowing)
                    new_data = []
                    modified = False
                    for count, (revnum, revision, timestamp) in enumerate(all_data[::-1]):
                        if count < MINIMUM_PERMANENT_CSETS:
                            if timestamp != -1:
                                modified = True
                                new_data.append((revnum, revision, -1))
                            else:
                                new_data.append((revnum, revision, timestamp))
                        elif type(timestamp) != int or timestamp == -1:
                            modified = True
                            new_data.append((revnum, revision, int(time.time())))
                        else:
                            new_data.append((revnum, revision, timestamp))

                    # Delete annotations at revisions with timestamps
                    # that are too old. The csetLog entries will have
                    # their timestamps reset here.
                    new_data1 = []
                    annrevs_to_del = []
                    current_time = time.time()
                    for count, (revnum, revision, timestamp) in enumerate(new_data[::-1]):
                        new_timestamp = timestamp
                        if timestamp != -1:
                            if current_time >= timestamp + TIME_TO_KEEP_ANNOTATIONS.seconds:
                                modified = True
                                new_timestamp = current_time
                                annrevs_to_del.append(revision)
                        new_data1.append((revnum, revision, new_timestamp))

                    if len(annrevs_to_del) > 0:
                        # Delete any latestFileMod and annotation entries
                        # that are too old.
                        Log.note(
                            "Deleting annotations and latestFileMod for revisions for being "
                            "older than {{oldest}}: {{revisions}}",
                            oldest=TIME_TO_KEEP_ANNOTATIONS,
                            revisions=annrevs_to_del
                        )
                        with self.conn.transaction() as t:
                            t.execute(
                                "DELETE FROM latestFileMod WHERE revision IN " +
                                quote_set(annrevs_to_del)
                            )
                            t.execute(
                                "DELETE FROM annotations WHERE revision IN " +
                                quote_set(annrevs_to_del)
                            )

                    # Delete any overflowing entries
                    new_data2 = new_data1
                    reved_all_data = all_data[::-1]
                    deleted_data = reved_all_data[MAXIMUM_NONPERMANENT_CSETS:]
                    delete_overflowing_revstart = None
                    if len(deleted_data) > 0:
                        _, delete_overflowing_revstart, _ = deleted_data[0]
                        new_data2 = set(all_data) - set(deleted_data)

                        # Update old frontiers if requested, otherwise
                        # they will all get deleted by the csetLog_deleter
                        # worker
                        if UPDATE_VERY_OLD_FRONTIERS:
                            _, max_revision, _ = all_data[-1]
                            for _, revision, _ in deleted_data:
                                with self.conn.transaction() as t:
                                    old_files = t.get(
                                        "SELECT file FROM latestFileMod WHERE revision=?",
                                        (revision,)
                                    )
                                if old_files is None or len(old_files) <= 0:
                                    continue

                                self.tuid_service.get_tuids_from_files(
                                    old_files,
                                    max_revision,
                                    going_forward=True,
                                )

                                still_exist = True
                                while still_exist and not please_stop:
                                    Till(seconds=TUID_EXISTENCE_WAIT_TIME).wait()
                                    with self.conn.transaction() as t:
                                        old_files = t.get(
                                            "SELECT file FROM latestFileMod WHERE revision=?",
                                            (revision,)
                                        )
                                    if old_files is None or len(old_files) <= 0:
                                        still_exist = False

                    # Update table and schedule a deletion
                    if modified:
                        with self.conn.transaction() as t:
                            t.execute(
                                "INSERT OR REPLACE INTO csetLog (revnum, revision, timestamp) VALUES " +
                                sql_list(
                                    quote_set(cset_entry)
                                    for cset_entry in new_data2
                                )
                            )
                    if not deleted_data:
                        continue

                    Log.note("Scheduling {{num_csets}} for deletion", num_csets=len(deleted_data))
                    self.deletions_todo.add(delete_overflowing_revstart)
            except Exception as e:
                Log.warning("Unexpected error occured while maintaining csetLog, continuing to try: ", cause=e)
        return