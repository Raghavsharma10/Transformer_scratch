def aggregate_per_as(self, start_time, end_time):
        """ Given a time range aggregates bytes per ASNs.

            Args:
                start_time: A string representing the starting time of the time range
                end_time: A string representing the ending time of the time range

            Returns:
                A list of prefixes sorted by sum_bytes. For example:

                [
                        {'key': '6500', 'sum_bytes': 3000},
                        {'key': '2310', 'sum_bytes': 2000},
                        {'key': '8182', 'sum_bytes': 1000},
                ]
        """

        query = ''' SELECT as_dst as key, SUM(bytes) as sum_bytes
                    from acct
                    WHERE
                    datetime(stamp_updated) BETWEEN datetime(?) AND datetime(?, "+1 second")
                    GROUP by as_dst
                    ORDER BY SUM(bytes) DESC;
                '''

        return self._execute_query(query, [start_time, end_time])