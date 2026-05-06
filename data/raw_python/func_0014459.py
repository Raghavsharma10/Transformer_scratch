def _get_date_facet_counts(self, timespan, date_field, start_date=None, end_date=None):
        '''
        Returns Range Facet counts based on
        '''
        if 'DAY' not in timespan:
            raise ValueError("At this time, only DAY date range increment is supported. Aborting..... ")

        #Need to do this a bit better later. Don't like the string and date concatenations.
        if not start_date:
            start_date = self._get_edge_date(date_field, 'asc')
            start_date = datetime.strptime(start_date,'%Y-%m-%dT%H:%M:%S.%fZ').date().isoformat()+'T00:00:00.000Z'
        else:
            start_date = start_date+'T00:00:00.000Z'

        if not end_date:
            end_date = self._get_edge_date(date_field, 'desc')
            end_date = datetime.strptime(end_date,'%Y-%m-%dT%H:%M:%S.%fZ').date()
            end_date += timedelta(days=1)
            end_date = end_date.isoformat()+'T00:00:00.000Z'
        else:
            end_date = end_date+'T00:00:00.000Z'


        self.log.info("Processing Items from {} to {}".format(start_date, end_date))

        #Get facet counts for source and destination collections
        source_facet = self._source.query(self._source_coll,
            self._get_date_range_query(timespan=timespan, start_date=start_date, end_date=end_date)
            ).get_facets_ranges()[date_field]
        dest_facet = self._dest.query(
            self._dest_coll, self._get_date_range_query(
                    timespan=timespan, start_date=start_date, end_date=end_date
                    )).get_facets_ranges()[date_field]
        return source_facet, dest_facet