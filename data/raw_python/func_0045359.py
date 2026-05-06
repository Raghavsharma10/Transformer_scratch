def _parse_query_modifier(self, modifier, qval, is_escaped):
        """
        Parses query_value according to query_type

        Args:
            modifier (str): Type of query. Exact, contains, lte etc.
            qval: Value partition of the query.

        Returns:
            Parsed query_value.
        """
        if modifier == 'range':
            if not qval[0]:
                start = '*'
            elif isinstance(qval[0], date):
                start = self._handle_date(qval[0])
            elif isinstance(qval[0], datetime):
                start = self._handle_datetime(qval[0])
            elif not is_escaped:
                start = self._escape_query(qval[0])
            else:
                start = qval[0]
            if not qval[1]:
                end = '*'
            elif isinstance(qval[1], date):
                end = self._handle_date(qval[1])
            elif isinstance(qval[1], datetime):
                end = self._handle_datetime(qval[1])
            elif not is_escaped:
                end = self._escape_query(qval[1])
            else:
                end = qval[1]
            qval = '[%s TO %s]' % (start, end)
        else:
            if not is_escaped and not isinstance(qval, (date, datetime, int, float)):
                qval = self._escape_query(qval)
            if modifier == 'exact':
                qval = qval
            elif modifier == 'contains':
                qval = "*%s*" % qval
            elif modifier == 'startswith':
                qval = "%s*" % qval
            elif modifier == 'endswith':
                qval = "%s*" % qval
            elif modifier == 'lte':
                qval = '[* TO %s]' % qval
            elif modifier == 'gte':
                qval = '[%s TO *]' % qval
            elif modifier == 'lt':
                if isinstance(qval, int):
                    qval -= 1
                qval = '[* TO %s]' % qval
            elif modifier == 'gt':
                if isinstance(qval, int):
                    qval += 1
                qval = '[%s TO *]' % qval
        return qval