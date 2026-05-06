def update_tip(self):
        '''
        Returns False if the tip is already at the newest, or True
        if an update has taken place.
        :return:
        '''
        clog_obj = self._get_clog(self.tuid_service.hg_url / self.config.hg.branch / 'json-log' / 'tip')

        # Get current tip in DB
        with self.conn.transaction() as t:
            _, newest_known_rev = self.get_tip(t)

        # If we are still at the newest, wait for CSET_TIP_WAIT_TIME seconds
        # before checking again.
        first_clog_entry = clog_obj['changesets'][0]['node'][:12]
        if newest_known_rev == first_clog_entry:
            return False

        csets_to_gather = None
        if not newest_known_rev:
            Log.note(
                "No revisions found in table, adding {{minim}} entries...",
                minim=MINIMUM_PERMANENT_CSETS
            )
            csets_to_gather = MINIMUM_PERMANENT_CSETS

        found_newest_known = False
        csets_to_add = []
        csets_found = 0
        clogs_seen = 0
        Log.note("Found new revisions. Updating csetLog tip to {{rev}}...", rev=first_clog_entry)
        while not found_newest_known and clogs_seen < MAX_TIPFILL_CLOGS:
            clog_csets_list = list(clog_obj['changesets'])
            for clog_cset in clog_csets_list[:-1]:
                nodes_cset = clog_cset['node'][:12]
                if not csets_to_gather:
                    if nodes_cset == newest_known_rev:
                        found_newest_known = True
                        break
                else:
                    if csets_found >= csets_to_gather:
                        found_newest_known = True
                        break
                csets_found += 1
                csets_to_add.append(nodes_cset)
            if not found_newest_known:
                # Get the next page
                clogs_seen += 1
                final_rev = clog_csets_list[-1]['node'][:12]
                clog_url = self.tuid_service.hg_url / self.config.hg.branch / 'json-log' / final_rev
                clog_obj = self._get_clog(clog_url)

        if clogs_seen >= MAX_TIPFILL_CLOGS:
            Log.error(
                "Too many changesets, can't find last tip or the number is too high: {{rev}}. "
                "Maximum possible to request is {{maxnum}}",
                rev=coalesce(newest_known_rev, csets_to_gather),
                maxnum=MAX_TIPFILL_CLOGS * CHANGESETS_PER_CLOG
            )
            return False

        with self.working_locker:
            Log.note("Adding {{csets}}", csets=csets_to_add)
            self.add_cset_entries(csets_to_add, timestamp=False)
        return True