def copy(self, keys, source, sample_only_filter=None, sample_size=None, done_copy=None):
        """
        :param keys: THE KEYS TO LOAD FROM source
        :param source: THE SOURCE (USUALLY S3 BUCKET)
        :param sample_only_filter: SOME FILTER, IN CASE YOU DO NOT WANT TO SEND EVERYTHING
        :param sample_size: FOR RANDOM SAMPLE OF THE source DATA
        :param done_copy: CALLBACK, ADDED TO queue, TO FINISH THE TRANSACTION
        :return: LIST OF SUB-keys PUSHED INTO ES
        """
        num_keys = 0
        queue = None
        pending = []  # FOR WHEN WE DO NOT HAVE QUEUE YET
        for key in keys:
            timer = Timer("Process {{key}}", param={"key": key}, silent=not DEBUG)
            try:
                with timer:
                    for rownum, line in enumerate(source.read_lines(strip_extension(key))):
                        if not line:
                            continue

                        if rownum > 0 and rownum % 1000 == 0:
                            Log.note("Ingested {{num}} records from {{key}} in bucket {{bucket}}", num=rownum, key=key, bucket=source.name)

                        insert_me, please_stop = fix(key, rownum, line, source, sample_only_filter, sample_size)
                        if insert_me == None:
                            continue
                        value = insert_me['value']

                        if '_id' not in value:
                            Log.warning("expecting an _id in all S3 records. If missing, there can be duplicates")

                        if queue == None:
                            queue = self._get_queue(insert_me)
                            if queue == None:
                                pending.append(insert_me)
                                if len(pending) > 1000:
                                    if done_copy:
                                        done_copy()
                                    Log.error("first 1000 (key={{key}}) records for {{alias}} have no indication what index to put data", key=tuple(keys)[0], alias=self.settings.index)
                                continue
                            elif queue is DATA_TOO_OLD:
                                break
                            if pending:
                                queue.extend(pending)
                                pending = []

                        num_keys += 1
                        queue.add(insert_me)

                        if please_stop:
                            break
            except Exception as e:
                if KEY_IS_WRONG_FORMAT in e:
                    Log.warning("Could not process {{key}} because bad format. Never trying again.", key=key, cause=e)
                    pass
                elif CAN_NOT_DECODE_JSON in e:
                    Log.warning("Could not process {{key}} because of bad JSON. Never trying again.", key=key, cause=e)
                    pass
                else:
                    Log.warning("Could not process {{key}} after {{duration|round(places=2)}}seconds", key=key, duration=timer.duration.seconds, cause=e)
                    done_copy = None

        if done_copy:
            if queue == None:
                done_copy()
            elif queue is DATA_TOO_OLD:
                done_copy()
            else:
                queue.add(done_copy)

        if [p for p in pending if wrap(p).value.task.state not in ('failed', 'exception')]:
            Log.error("Did not find an index for {{alias}} to place the data for key={{key}}", key=tuple(keys)[0], alias=self.settings.index)

        Log.note("{{num}} keys from {{key|json}} added", num=num_keys, key=keys)
        return num_keys