def _activity_import_doc(self, time_doc, activities):
        '''
        Import activities for a single document into timeline.
        '''
        batch_updates = [time_doc]
        # We want to consider only activities that happend before time_doc
        # do not move this, because time_doc._start changes
        # time_doc['_start'] is a timestamp, whereas act[0] is a datetime
        # we need to be sure to convert act[0] (when) to timestamp!
        td_start = time_doc['_start']
        activities = filter(lambda act: (act[0] < td_start and
                                         act[1] in time_doc), activities)
        creation_field = self.lconfig.get('cfield')
        # make sure that activities are sorted by when descending
        activities.sort(reverse=True, key=lambda o: o[0])
        new_doc = {}
        for when, field, removed, added in activities:
            last_doc = batch_updates.pop()
            # check if this activity happened at the same time as the last one,
            # if it did then we need to group them together
            if last_doc['_end'] == when:
                new_doc = deepcopy(last_doc)
                last_doc = batch_updates.pop()
            else:
                new_doc = deepcopy(last_doc)
                new_doc['_start'] = when
                new_doc['_end'] = when
                last_doc['_start'] = when
            last_val = last_doc[field]

            new_val, inconsistent = self._activity_backwards(new_doc[field],
                                                             removed, added)
            new_doc[field] = new_val
            # Check if the object has the correct field value.
            if inconsistent:
                self._log_inconsistency(last_doc, last_val, field,
                                        removed, added, when)
                new_doc['_e'] = {} if not new_doc.get('_e') else new_doc['_e']
                # set curreupted field value to the the value that was added
                # and continue processing as if that issue didn't exist
                new_doc['_e'][field] = added
            # Add the objects to the batch
            batch_updates.extend([last_doc, new_doc])
        # try to set the _start of the first version to the creation time
        try:
            # set start to creation time if available
            last_doc = batch_updates[-1]
            if creation_field:
                # again, we expect _start to be epoch float...
                creation_ts = dt2ts(last_doc[creation_field])
                if creation_ts < last_doc['_start']:
                    last_doc['_start'] = creation_ts
                elif len(batch_updates) == 1:
                    # we have only one version, that we did not change
                    return []
                else:
                    pass  # leave as-is
        except Exception as e:
            logger.error('Error updating creation time; %s' % e)
        return batch_updates