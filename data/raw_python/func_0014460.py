def resume(self, start_date=None, end_date=None, timespan='DAY', check= False):
        '''
        This method may help if the original run was interrupted for some reason. It will only work under the following conditions
        * You have a date field that you can facet on
        * Indexing was stopped for the duration of the copy

        The way this tries to resume re-indexing is by running a date range facet on the source and destination collections. It then compares
        the counts in both collections for each timespan specified. If the counts are different, it will re-index items for each range where
        the counts are off. You can also pass in a start_date to only get items after a certain time period. Note that each date range will be indexed in
        it's entirety, even if there is only one item missing.

        Keep in mind this only checks the counts and not actual data. So make the indexes weren't modified between the reindexing execution and
        running the resume operation.

        :param start_date: Date to start indexing from. If not specified there will be no restrictions and all data will be processed. Note that
        this value will be passed to Solr directly and not modified.
        :param end_date: The date to index items up to. Solr Date Math compliant value for faceting; currenlty only DAY is supported.
        :param timespan: Solr Date Math compliant value for faceting; currenlty only DAY is supported.
        :param check: If set to True it will only log differences between the two collections without actually modifying the destination.
        '''

        if type(self._source) is not SolrClient or type(self._dest) is not SolrClient:
            raise ValueError("To resume, both source and destination need to be Solr.")

        source_facet, dest_facet = self._get_date_facet_counts(timespan, self._date_field, start_date=start_date, end_date=end_date)

        for dt_range in sorted(source_facet):
            if dt_range in dest_facet:
                self.log.info("Date Range: {} Source: {} Destination:{} Difference:{}".format(
                        dt_range, source_facet[dt_range], dest_facet[dt_range], (source_facet[dt_range]-dest_facet[dt_range])))
                if check:
                    continue
                if source_facet[dt_range] > dest_facet[dt_range]:
                    #Kicks off reindexing with an additional FQ
                    self.reindex(fq=['{}:[{} TO {}]'.format(self._date_field, dt_range, dt_range+'+1{}'.format(timespan))])
                    self.log.info("Complete Date Range {}".format(dt_range))
            else:
                self.log.error("Something went wrong; destinationSource: {}".format(source_facet))
                self.log.error("Destination: {}".format(dest_facet))
                raise ValueError("Date Ranges don't match up")
        self._dest.commit(self._dest_coll, openSearcher=True)