def _get_tuids(
            self,
            transaction,
            files,
            revision,
            annotated_files,
            commit=True,
            repo=None
        ):
        '''
        Returns (TUID, line) tuples for a given file at a given revision.

        Uses json-annotate to find all lines in this revision, then it updates
        the database with any missing revisions for the file changes listed
        in annotate. Then, we use the information from annotate coupled with the
        diff information that was inserted into the DB to return TUIDs. This way
        we don't have to deal with child, parents, dates, etc..

        :param files: list of files to process
        :param revision: revision at which to get the file
        :param annotated_files: annotations for each file
        :param commit: True to commit new TUIDs else False
        :param repo: The branch to get tuids from
        :return: List of TuidMap objects
        '''
        results = []

        for fcount, annotated_object in enumerate(annotated_files):
            file = files[fcount]

            # TODO: Replace old empty annotation if a new one is found
            # TODO: at the same revision and if it is not empty as well.
            # Make sure we are not adding the same thing another thread
            # added.
            tmp_ann = self._get_annotation(revision, file, transaction=transaction)
            if tmp_ann != None:
                results.append((file, self.destringify_tuids(tmp_ann)))
                continue

            # If it's not defined at this revision, we need to add it in
            errored = False
            if isinstance(annotated_object, (text_type, str)):
                errored = True
                Log.warning(
                    "{{file}} does not exist in the revision={{cset}} branch={{branch_name}}",
                    branch_name=repo,
                    cset=revision,
                    file=file
                )
            elif annotated_object is None:
                Log.warning(
                    "Unexpected error getting annotation for: {{file}} in the revision={{cset}} branch={{branch_name}}",
                    branch_name=repo,
                    cset=revision,
                    file=file
                )
                errored = True
            elif 'annotate' not in annotated_object:
                Log.warning(
                    "Missing annotate, type got: {{ann_type}}, expecting:dict returned when getting "
                    "annotation for: {{file}} in the revision {{cset}}",
                    cset=revision, file=file, ann_type=type(annotated_object)
                )
                errored = True

            if errored:
                Log.note("Inserting dummy entry...")
                self.insert_tuid_dummy(transaction, revision, file, commit=commit)
                self.insert_annotate_dummy(transaction, revision, file, commit=commit)
                results.append((file, []))
                continue

            # Gather all missing csets and the
            # corresponding lines.
            line_origins = []
            for node in annotated_object['annotate']:
                cset_len12 = node['node'][:12]

                # If the line added by `cset_len12` is not known
                # add it. Use the 'abspath' field to determine the
                # name of the file it was created in (in case it was
                # changed).
                line_origins.append((node['abspath'], cset_len12, int(node['targetline'])))

            file_names = list(set([f for f, _, _ in line_origins]))
            revs_to_find = list(set([rev for _, rev, _ in line_origins]))
            lines_to_find = list(set([line for _, _, line in line_origins]))
            existing_tuids_tmp = {
                str((file, revision, line)): tuid
                for tuid, file, revision, line in transaction.query(
                    "SELECT tuid, file, revision, line FROM temporal"
                    " WHERE file IN " + quote_list(file_names) +
                    " AND revision IN " + quote_list(revs_to_find) +
                    " AND line IN " + quote_list(lines_to_find)
                ).data
            }

            # Recompute existing tuids based on line_origins
            # entry ordering because we can't order them any other way
            # since the `line` entry in the `temporal` table is relative
            # to it's creation date, not the currently requested
            # annotation.
            existing_tuids = {
                (line_num+1): existing_tuids_tmp[str(ann_entry)]
                for line_num, ann_entry in enumerate(line_origins)
                if str(ann_entry) in existing_tuids_tmp
            }
            new_lines = set([line_num+1 for line_num, _ in enumerate(line_origins)]) - set(existing_tuids.keys())

            # Update DB with any revisions found in annotated
            # object that are not in the DB.
            new_line_origins = {}
            if len(new_lines) > 0:
                try:
                    '''
                        HG Annotate Bug, Issue #58:
                        Here is where we assign the new tuids for the first
                        time we see duplicate entries - they are left
                        in `new_line_origins` after duplicates are found.
                        We only remove it from the lines to insert. In future
                        requests, `existing_tuids` above will handle duplicating
                        tuids for the entries if needed.
                    '''
                    new_line_origins = {
                        line_num: (self.tuid(),) + line_origins[line_num - 1]
                        for line_num in new_lines
                    }

                    duplicate_lines = {
                        line_num+1: line
                        for line_num, line in enumerate(line_origins)
                        if line in line_origins[:line_num]
                    }
                    if len(duplicate_lines) > 0:
                        Log.note(
                            "Duplicates found in {{file}} at {{cset}}: {{dupes}}",
                            file=file,
                            cset=revision,
                            dupes=str(duplicate_lines)
                        )
                        lines_to_insert = [
                            line
                            for line_num, line in new_line_origins.items()
                            if line_num not in duplicate_lines
                        ]
                    else:
                        lines_to_insert = new_line_origins.values()

                    for _, part_of_insert in jx.groupby(lines_to_insert, size=SQL_BATCH_SIZE):
                        transaction.execute(
                            "INSERT INTO temporal (tuid, file, revision, line)"
                            " VALUES " +
                            sql_list(
                                sql_iso(
                                    sql_list(map(quote_value, (tuid, f, rev, line_num)))
                                ) for tuid, f, rev, line_num in list(part_of_insert)
                            )
                        )

                    # Format so we don't have to use [0] to get at the tuid
                    new_line_origins = {line_num: new_line_origins[line_num][0] for line_num in new_line_origins}
                except Exception as e:
                    # Something broke for this file, ignore it and go to the
                    # next one.
                    Log.note("Failed to insert new tuids {{cause}}", cause=e)
                    continue

            tuids = []
            for line_ind, line_origin in enumerate(line_origins):
                line_num = line_ind + 1
                if line_num in existing_tuids:
                    tuids.append(TuidMap(existing_tuids[line_num], line_num))
                else:
                    tuids.append(TuidMap(new_line_origins[line_num], line_num))

            self.insert_annotations(
                transaction,
                [(
                    revision,
                    file,
                    self.stringify_tuids(tuids)
                )]
            )
            results.append((file, tuids))

        return results