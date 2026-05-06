def _apply_diff(self, transaction, annotation, diff, cset, file):
        '''
        Using an annotation ([(tuid,line)] - array
        of TuidMap objects), we change the line numbers to
        reflect a given diff and return them. diff must
        be a diff object returned from get_diff(cset, file).
        Only for going forward in time, not back.

        :param annotation: list of TuidMap objects
        :param diff: unified diff from get_diff
        :param cset: revision to apply diff at
        :param file: name of file diff is applied to
        :return:
        '''
        # Ignore merges, they have duplicate entries.
        if diff['merge']:
            return annotation, file
        if file.lstrip('/') == 'dev/null':
            return [], file

        list_to_insert = []
        new_ann = [x for x in annotation]
        new_ann.sort(key=lambda x: x.line)

        def add_one(tl_tuple, lines):
            start = tl_tuple.line
            return lines[:start-1] + [tl_tuple] + [TuidMap(tmap.tuid, int(tmap.line) + 1) for tmap in lines[start-1:]]

        def remove_one(start, lines):
            return lines[:start-1] + [TuidMap(tmap.tuid, int(tmap.line) - 1) for tmap in lines[start:]]

        for f_proc in diff['diffs']:
            new_fname = f_proc['new'].name.lstrip('/')
            old_fname = f_proc['old'].name.lstrip('/')
            if new_fname != file and old_fname != file:
                continue
            if old_fname != new_fname:
                if new_fname == 'dev/null':
                    return [], file
                # Change the file name so that new tuids
                # are correctly created.
                file = new_fname

            f_diff = f_proc['changes']
            for change in f_diff:
                if change.action == '+':
                    tuid_tmp = self._get_one_tuid(transaction, cset, file, change.line+1)
                    if not tuid_tmp:
                        new_tuid = self.tuid()
                        list_to_insert.append((new_tuid, cset, file, change.line+1))
                    else:
                        new_tuid = tuid_tmp[0]
                    new_ann = add_one(TuidMap(new_tuid, change.line+1), new_ann)
                elif change.action == '-':
                    new_ann = remove_one(change.line+1, new_ann)
            break # Found the file, exit searching

        if len(list_to_insert) > 0:
            count = 0
            for _, inserts_list in jx.groupby(list_to_insert, size=SQL_BATCH_SIZE):
                transaction.execute(
                    "INSERT INTO temporal (tuid, revision, file, line)"
                    " VALUES " +
                    sql_list(quote_list(tp) for tp in inserts_list)
                )

        return new_ann, file