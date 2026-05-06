def csetLog_deleter(self, please_stop=None):
        '''
        Deletes changesets from the csetLog table
        and also changesets from the annotation table
        that have revisions matching the given changesets.
        Accepts lists of csets from self.deletions_todo.
        :param please_stop:
        :return:
        '''
        while not please_stop:
            try:
                request = self.deletions_todo.pop(till=please_stop)
                if please_stop:
                    break

                # If deletion is disabled, ignore the current
                # request - it will need to be re-requested.
                if self.disable_deletion:
                    Till(till=CSET_DELETION_WAIT_TIME).wait()
                    continue

                with self.working_locker:
                    first_cset = request

                    # Since we are deleting and moving stuff around in the
                    # TUID tables, we need everything to be contained in
                    # one transaction with no interruptions.
                    with self.conn.transaction() as t:
                        revnum = self._get_one_revnum(t, first_cset)[0]
                        csets_to_del = t.get(
                            "SELECT revnum, revision FROM csetLog WHERE revnum <= ?", (revnum,)
                        )
                        csets_to_del = [cset for _, cset in csets_to_del]
                        existing_frontiers = t.query(
                            "SELECT revision FROM latestFileMod WHERE revision IN " +
                            quote_set(csets_to_del)
                        ).data

                        existing_frontiers = [existing_frontiers[i][0] for i, _ in enumerate(existing_frontiers)]
                        Log.note(
                            "Deleting all annotations and changeset log entries with revisions in the list: {{csets}}",
                            csets=csets_to_del
                        )

                        if len(existing_frontiers) > 0:
                            # This handles files which no longer exist anymore in
                            # the main branch.
                            Log.note(
                                "Deleting existing frontiers for revisions: {{revisions}}",
                                revisions=existing_frontiers
                            )
                            t.execute(
                                "DELETE FROM latestFileMod WHERE revision IN " +
                                quote_set(existing_frontiers)
                            )

                        Log.note("Deleting annotations...")
                        t.execute(
                            "DELETE FROM annotations WHERE revision IN " +
                            quote_set(csets_to_del)
                        )

                        Log.note(
                            "Deleting {{num_entries}} csetLog entries...",
                            num_entries=len(csets_to_del)
                        )
                        t.execute(
                            "DELETE FROM csetLog WHERE revision IN " +
                            quote_set(csets_to_del)
                        )

                    # Recalculate the revnums
                    self.recompute_table_revnums()
            except Exception as e:
                Log.warning("Unexpected error occured while deleting from csetLog:", cause=e)
                Till(seconds=CSET_DELETION_WAIT_TIME).wait()
        return