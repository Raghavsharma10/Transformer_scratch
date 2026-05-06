def _get_tuids_from_files_try_branch(self, files, revision):
        '''
        Gets files from a try revision. It abuses the idea that try pushes
        will come from various, but stable points (if people make many
        pushes on that revision). Furthermore, updates are generally done
        to a revision that should eventually have tuids already in the DB
        (i.e. overtime as people update to revisions that have a tuid annotation).

        :param files: Files to query.
        :param revision: Revision to get them at.
        :return: List of (file, tuids) tuples.
        '''

        repo = 'try'
        result = []
        log_existing_files = []
        files_to_update = []

        # Check if the files were already annotated.
        for file in files:
            with self.conn.transaction() as t:
                already_ann = self._get_annotation(revision, file, transaction=t)
            if already_ann:
                result.append((file, self.destringify_tuids(already_ann)))
                log_existing_files.append('exists|' + file)
                continue
            elif already_ann[0] == '':
                result.append((file, []))
                log_existing_files.append('removed|' + file)
                continue
            else:
                files_to_update.append(file)

        if len(log_existing_files) > 0:
            Log.note(
                "Try revision run - existing entries: {{count}}/{{total}} | {{percent}}",
                count=str(len(log_existing_files)),
                total=str(len(files)),
                percent=str(100*(len(log_existing_files)/len(files)))
            )

        if len(files_to_update) <= 0:
            Log.note("Found all files for try revision request: {{cset}}", cset=revision)
            return result

        # There are files to process, so let's find all the diffs.
        found_mc_patch = False
        diffs_to_get = [] # Will contain diffs in reverse order of application
        curr_rev = revision
        mc_revision = ''
        while not found_mc_patch:
            jsonrev_url = self.hg_url / repo / 'json-rev' / curr_rev
            try:
                Log.note("Searching through changelog {{url}}", url=jsonrev_url)
                clog_obj = http.get_json(jsonrev_url, retry=RETRY)
                if isinstance(clog_obj, (text_type, str)):
                    Log.error(
                        "Revision {{cset}} does not exist in the {{branch}} branch",
                        cset=curr_rev, branch=repo
                    )
                if 'phase' not in clog_obj:
                    Log.warning(
                        "Unexpected error getting changset-log for {{url}}: `phase` entry cannot be found.",
                        url=jsonrev_url
                    )
                    return [(file, []) for file in files]
            except Exception as e:
                Log.warning(
                    "Unexpected error getting changset-log for {{url}}: {{error}}",
                    url=jsonrev_url, error=e
                )
                return [(file, []) for file in files]

            # When `phase` is public, the patch is (assumed to be)
            # in any repo other than try.
            if clog_obj['phase'] == 'public':
                found_mc_patch = True
                mc_revision = curr_rev
                continue
            elif clog_obj['phase'] == 'draft':
                diffs_to_get.append(curr_rev)
            else:
                Log.warning(
                    "Unknown `phase` state `{{state}}` encountered at revision {{cset}}",
                    cset=curr_rev, state=clog_obj['phase']
                )
                return [(file, []) for file in files]
            curr_rev = clog_obj['parents'][0][:12]

        added_files = {}
        removed_files = {}
        files_to_process = {}

        Log.note("Gathering diffs for: {{csets}}", csets=str(diffs_to_get))
        all_diffs = self.get_diffs(diffs_to_get, repo=repo)

        # Build a dict for faster access to the diffs
        parsed_diffs = {entry['cset']: entry['diff'] for entry in all_diffs}
        for csets_diff in all_diffs:
            cset_len12 = csets_diff['cset']
            parsed_diff = csets_diff['diff']['diffs']

            for f_added in parsed_diff:
                # Get new entries for removed files.
                new_name = f_added['new'].name.lstrip('/')
                old_name = f_added['old'].name.lstrip('/')

                # If we don't need this file, skip it
                if new_name not in files_to_update:
                    # If the file was removed, set a
                    # flag and return no tuids later.
                    if new_name == 'dev/null':
                        removed_files[old_name] = True
                    continue

                if old_name == 'dev/null':
                    added_files[new_name] = True
                    continue

                if new_name in files_to_process:
                    files_to_process[new_name].append(cset_len12)
                else:
                    files_to_process[new_name] = [cset_len12]

        # We've found a good patch (a public one), get it
        # for all files and apply the patch's onto it.
        curr_annotations = self.get_tuids(files, mc_revision, commit=False)
        curr_annots_dict = {file: mc_annot for file, mc_annot in curr_annotations}

        anns_to_get = []
        ann_inserts = []
        tmp_results = {}

        with self.conn.transaction() as transaction:
            for file in files_to_update:
                if file not in curr_annots_dict:
                    Log.note(
                        "WARNING: Missing annotation entry in mozilla-central branch revision {{cset}} "
                        "for {{file}}",
                        file=file, cset=mc_revision
                    )
                    # Try getting it from the try revision
                    anns_to_get.append(file)
                    continue

                if file in added_files:
                    Log.note("Try revision run - added: {{file}}", file=file)
                    anns_to_get.append(file)
                elif file in removed_files:
                    Log.note("Try revision run - removed: {{file}}", file=file)
                    ann_inserts.append((revision, file, ''))
                    tmp_results[file] = []
                elif file in files_to_process:
                    # Reverse the list, we always find the newest diff first
                    Log.note("Try revision run - modified: {{file}}", file=file)
                    csets_to_proc = files_to_process[file][::-1]
                    old_ann = curr_annots_dict[file]

                    # Apply all the diffs
                    tmp_res = old_ann
                    new_fname = file
                    for i in csets_to_proc:
                        tmp_res, new_fname = self._apply_diff(transaction, tmp_res, parsed_diffs[i], i, new_fname)

                    ann_inserts.append((revision, file, self.stringify_tuids(tmp_res)))
                    tmp_results[file] = tmp_res
                else:
                    # Nothing changed with the file, use it's current annotation
                    Log.note("Try revision run - not modified: {{file}}", file=file)
                    ann_inserts.append((revision, file, self.stringify_tuids(curr_annots_dict[file])))
                    tmp_results[file] = curr_annots_dict[file]

            # Insert and check annotations, get all that were
            # added by another thread.
            anns_added_by_other_thread = {}
            if len(ann_inserts) > 0:
                ann_inserts = list(set(ann_inserts))
                for _, tmp_inserts in jx.groupby(ann_inserts, size=SQL_ANN_BATCH_SIZE):
                    # Check if any were added in the mean time by another thread
                    recomputed_inserts = []
                    for rev, filename, tuids in tmp_inserts:
                        tmp_ann = self._get_annotation(rev, filename, transaction=transaction)
                        if not tmp_ann:
                            recomputed_inserts.append((rev, filename, tuids))
                        else:
                            anns_added_by_other_thread[filename] = self.destringify_tuids(tmp_ann)

                    try:
                        self.insert_annotations(transaction, recomputed_inserts)
                    except Exception as e:
                        Log.error("Error inserting into annotations table.", cause=e)

        if len(anns_to_get) > 0:
            result.extend(self.get_tuids(anns_to_get, revision, repo=repo))

        for f in tmp_results:
            tuids = tmp_results[f]
            if f in anns_added_by_other_thread:
                tuids = anns_added_by_other_thread[f]
            result.append((f, tuids))
        return result