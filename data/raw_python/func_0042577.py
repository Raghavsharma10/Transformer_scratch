def add_cset_entries(self, ordered_rev_list, timestamp=False, number_forward=True):
        '''
        Adds a list of revisions to the table. Assumes ordered_rev_list is an ordered
        based on how changesets are found in the changelog. Going forwards or backwards is dealt
        with by flipping the list
        :param ordered_cset_list: Order given from changeset log searching.
        :param timestamp: If false, records are kept indefinitely
                          but if holes exist: (delete, None, delete, None)
                          those delete's with None's around them
                          will not be deleted.
        :param numbered: If True, this function will number the revision list
                         by going forward from max(revNum), else it'll go backwards
                         from revNum, then add X to all revnums and self.next_revnum
                         where X is the length of ordered_rev_list
        :return:
        '''
        with self.conn.transaction() as t:
            current_min = t.get_one("SELECT min(revnum) FROM csetlog")[0]
            current_max = t.get_one("SELECT max(revnum) FROM csetlog")[0]
            if not current_min or not current_max:
                current_min = 0
                current_max = 0

            direction = -1
            start = current_min - 1
            if number_forward:
                direction = 1
                start = current_max + 1
                ordered_rev_list = ordered_rev_list[::-1]

            insert_list = [
                (
                    start + direction * count,
                    rev,
                    int(time.time()) if timestamp else -1
                )
                for count, rev in enumerate(ordered_rev_list)
            ]

            # In case of overlapping requests
            fmt_insert_list = []
            for cset_entry in insert_list:
                tmp = self._get_one_revision(t, cset_entry)
                if not tmp:
                    fmt_insert_list.append(cset_entry)

            for _, tmp_insert_list in jx.groupby(fmt_insert_list, size=SQL_CSET_BATCH_SIZE):
                t.execute(
                    "INSERT INTO csetLog (revnum, revision, timestamp)" +
                    " VALUES " +
                    sql_list(
                        quote_set((revnum, revision, timestamp))
                        for revnum, revision, timestamp in tmp_insert_list
                    )
                )

            # Move the revision numbers forward if needed
            self.recompute_table_revnums()

        # Start a maintenance run if needed
        if self.check_for_maintenance():
            self.maintenance_signal.go()