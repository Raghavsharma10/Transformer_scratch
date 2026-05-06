def _daemon(self, please_stop, only_coverage_revisions=False):
        '''
        Runs continuously to prefill the temporal and
        annotations table with the coverage revisions*.

        * A coverage revision is a revision which has had
        code coverage run on it.

        :param please_stop: Used to stop the daemon
        :return: None
        '''
        while not please_stop:
            # Get all known files and their latest revisions on the frontier
            files_n_revs = self.conn.get("SELECT file, revision FROM latestFileMod")

            # Split these files into groups of revisions to make it
            # easier to update them. If we group them together, we
            # may end up updating groups that are new back to older
            # revisions.
            revs = {rev: [] for rev in set([file_n_rev[1] for file_n_rev in files_n_revs])}
            for file_n_rev in files_n_revs:
                revs[file_n_rev[1]].append(file_n_rev[0])

            # Go through each frontier and update it
            ran_changesets = False
            coverage_revisions = None
            for frontier in revs:
                if please_stop:
                    return

                files = revs[frontier]

                # Go through changeset logs until we find the last
                # known frontier for this revision group.
                csets = []
                final_rev = ''
                found_last_frontier = False
                Log.note("Searching for frontier: {{frontier}} ", frontier=frontier)
                Log.note("HG URL: {{url}}", url=self.hg_url / self.config.hg.branch / 'rev' / frontier)
                while not found_last_frontier:
                    # Get a changelog
                    clog_url = self.hg_url / self.config.hg.branch / 'json-log' / final_rev
                    try:
                        clog_obj = http.get_json(clog_url, retry=RETRY)
                    except Exception as e:
                        Log.error("Unexpected error getting changset-log for {{url}}", url=clog_url, error=e)

                    cset = ''
                    still_looking = True
                    # For each changeset/node
                    for clog_cset in clog_obj['changesets']:
                        cset = clog_cset['node'][:12]
                        if cset == frontier:
                            still_looking = False
                            break
                        csets.append(cset)

                    if not still_looking:
                        found_last_frontier = True
                    final_rev = cset

                # No csets found means that we are already
                # at the latest revisions.
                if len(csets) == 0:
                    continue

                # Get all the latest ccov and jsdcov revisions
                if (not coverage_revisions) and only_coverage_revisions:
                    active_data_url = 'http://activedata.allizom.org/query'
                    query_json = {
                        "limit": 1000,
                        "from": "task",
                        "where": {"and": [
                            {"in": {"build.type": ["ccov", "jsdcov"]}},
                            {"gte": {"run.timestamp": {"date": "today-day"}}},
                            {"eq": {"repo.branch.name": self.config.hg.branch}}
                        ]},
                        "select": [
                            {"aggregate": "min", "value": "run.timestamp"},
                            {"aggregate": "count"}
                        ],
                        "groupby": ["repo.changeset.id12"]
                    }
                    coverage_revisions_resp = http.post_json(active_data_url, retry=RETRY, data=query_json)
                    coverage_revisions = [rev_arr[0] for rev_arr in coverage_revisions_resp.data]

                # Reverse changeset list and for each code coverage revision
                # found by going through the list from oldest to newest,
                # update _all known_ file frontiers to that revision.
                csets.reverse()
                prev_cset = frontier
                for cset in csets:
                    if please_stop:
                        return
                    if only_coverage_revisions:
                        if cset not in coverage_revisions:
                            continue
                    if DEBUG:
                        Log.note("Moving frontier {{frontier}} forward to {{cset}}.", frontier=prev_cset, cset=cset)

                    # Update files
                    self.get_tuids_from_files(files, cset, going_forward=True)

                    ran_changesets = True
                    prev_cset = cset

            if not ran_changesets:
                (please_stop | Till(seconds=DAEMON_WAIT_AT_NEWEST.seconds)).wait()