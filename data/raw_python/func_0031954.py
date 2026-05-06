def search_command_record(
            self,
            after_context, before_context, context, context_type,
            **kwds):
        """
        Search command history.

        :rtype: [CommandRecord]

        """
        if after_context or before_context or context:
            kwds['condition_as_column'] = True
            limit = kwds['limit']
            kwds['limit'] = -1
            kwds['unique'] = False
            kwds['sort_by'] = {
                'session': ['session_start_time', 'start_time'],
                'time': ['start_time'],
            }[context_type]
            if not kwds['reverse']:
                # Default (reverse=False) means latest history comes first.
                after_context, before_context = before_context, after_context

        (sql, params, keys) = self._compile_sql_search_command_record(**kwds)
        records = self._select_rows(CommandRecord, keys, sql, params)

        # SOMEDAY: optimize context search;  do not create CommandRecord
        #          object for all (including non-matching) records.
        predicate = lambda r: r.condition
        if context:
            records = include_context(predicate, context, records)
        elif before_context:
            records = include_before(predicate, before_context, records)
        elif after_context:
            records = include_after(predicate, after_context, records)
        if after_context or before_context or context and limit >= 0:
            records = itertools.islice(records, limit)
        # NOTE: as SQLite does not support row_number function, let's
        #       do the filtering at Python side when context modifier
        #       is given.  This is *very* inefficient but at least it
        #       works..

        return records