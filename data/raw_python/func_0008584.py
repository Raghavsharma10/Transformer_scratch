def aggregate_per_prefix(self, start_time, end_time, limit=0, net_masks='', exclude_net_masks=False, filter_proto=None):
        """ Given a time range aggregates bytes per prefix.

            Args:
                start_time: A string representing the starting time of the time range
                end_time: A string representing the ending time of the time range
                limit: An optional integer. If it's >0 it will limit the amount of prefixes returned.
                filter_proto: Can be:
                    - None: Returns both ipv4 and ipv6
                    - 4: Returns only ipv4
                    - 6: Retruns only ipv6

            Returns:
                A list of prefixes sorted by sum_bytes. For example:

                [
                        {'key': '192.168.1.0/25', 'sum_bytes': 3000, 'as_dst': 345},
                        {'key': '192.213.1.0/25', 'sum_bytes': 2000, 'as_dst': 123},
                        {'key': '231.168.1.0/25', 'sum_bytes': 1000, 'as_dst': 321},
                ]
        """
        if net_masks == '':
            net_mask_filter = ''
        elif not exclude_net_masks:
            net_mask_filter = 'AND mask_dst IN ({})'.format(net_masks)
        elif exclude_net_masks:
            net_mask_filter = 'AND mask_dst NOT IN ({})'.format(net_masks)

        if filter_proto is None:
            proto_filter = ''
        elif int(filter_proto) == 4:
            proto_filter = 'AND ip_dst NOT LIKE "%:%"'
        elif int(filter_proto) == 6:
            proto_filter = 'AND ip_dst LIKE "%:%"'

        query = ''' SELECT ip_dst||'/'||mask_dst as key, SUM(bytes) as sum_bytes, as_dst
                    from acct
                    WHERE
                    datetime(stamp_updated) BETWEEN datetime(?) AND datetime(?, "+1 second")
                    {}
                    {}
                    GROUP by ip_dst,mask_dst
                    ORDER BY SUM(bytes) DESC
                '''.format(net_mask_filter, proto_filter)

        if limit > 0:
            query += 'LIMIT %d' % limit

        return self._execute_query(query, [start_time, end_time])