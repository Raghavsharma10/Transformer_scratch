def _fill_in_range(self, parent_cset, child_cset, timestamp=False, number_forward=True):
        '''
        Fills cset logs in a certain range. 'parent_cset' can be an int and in that case,
        we get that many changesets instead. If parent_cset is an int, then we consider
        that we are going backwards (number_forward is False) and we ignore the first
        changeset of the first log, and we ignore the setting for number_forward.
        Otherwise, we continue until we find the given 'parent_cset'.
        :param parent_cset:
        :param child_cset:
        :param timestamp:
        :param number_forward:
        :return:
        '''
        csets_to_add = []
        found_parent = False
        find_parent = False
        if type(parent_cset) != int:
            find_parent = True
        elif parent_cset >= MAX_BACKFILL_CLOGS * CHANGESETS_PER_CLOG:
            Log.warning(
                "Requested number of new changesets {{num}} is too high. "
                "Max number that can be requested is {{maxnum}}.",
                num=parent_cset,
                maxnum=MAX_BACKFILL_CLOGS * CHANGESETS_PER_CLOG
            )
            return None

        csets_found = 0
        clogs_seen = 0
        final_rev = child_cset
        while not found_parent and clogs_seen < MAX_BACKFILL_CLOGS:
            clog_url = self.tuid_service.hg_url / self.config.hg.branch / 'json-log' / final_rev
            clog_obj = self._get_clog(clog_url)
            clog_csets_list = list(clog_obj['changesets'])
            for clog_cset in clog_csets_list[:-1]:
                if not number_forward and csets_found <= 0:
                    # Skip this entry it already exists
                    csets_found += 1
                    continue

                nodes_cset = clog_cset['node'][:12]
                if find_parent:
                    if nodes_cset == parent_cset:
                        found_parent = True
                        if not number_forward:
                            # When going forward this entry is
                            # the given parent
                            csets_to_add.append(nodes_cset)
                        break
                else:
                    if csets_found + 1 > parent_cset:
                        found_parent = True
                        if not number_forward:
                            # When going forward this entry is
                            # the given parent (which is supposed
                            # to already exist)
                            csets_to_add.append(nodes_cset)
                        break
                    csets_found += 1
                csets_to_add.append(nodes_cset)
            if found_parent == True:
                break

            clogs_seen += 1
            final_rev = clog_csets_list[-1]['node'][:12]

        if found_parent:
            self.add_cset_entries(csets_to_add, timestamp=timestamp, number_forward=number_forward)
        else:
            Log.warning(
                "Couldn't find the end of the request for {{request}}. "
                "Max number that can be requested through _fill_in_range is {{maxnum}}.",
                request={
                    'parent_cset': parent_cset,
                    'child_cset':child_cset,
                    'number_forward': number_forward
                },
                maxnum=MAX_BACKFILL_CLOGS * CHANGESETS_PER_CLOG
            )
            return None
        return csets_to_add