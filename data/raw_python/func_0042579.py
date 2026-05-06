def fill_backward_with_list(self, please_stop=None):
        '''
        Expects requests of the tuple form: (parent_cset, timestamp)
        parent_cset can be an int X to go back by X changesets, or
        a string to search for going backwards in time. If timestamp
        is false, no timestamps will be added to the entries.
        :param please_stop:
        :return:
        '''
        while not please_stop:
            try:
                request = self.csets_todo_backwards.pop(till=please_stop)
                if please_stop:
                    break

                # If backfilling is disabled, all requests
                # are ignored.
                if self.disable_backfilling:
                    Till(till=CSET_BACKFILL_WAIT_TIME).wait()
                    continue

                if request:
                    parent_cset, timestamp = request
                else:
                    continue

                with self.working_locker:
                    with self.conn.transaction() as t:
                        parent_revnum = self._get_one_revnum(t, parent_cset)
                    if parent_revnum:
                        continue

                    with self.conn.transaction() as t:
                        _, oldest_revision = self.get_tail(t)

                    self._fill_in_range(
                        parent_cset,
                        oldest_revision,
                        timestamp=timestamp,
                        number_forward=False
                    )
                Log.note("Finished {{cset}}", cset=parent_cset)
            except Exception as e:
                Log.warning("Unknown error occurred during backfill: ", cause=e)