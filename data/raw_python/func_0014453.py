def _from_solr(self, fq=[], report_frequency = 25):
        '''
        Method for retrieving batch data from Solr.
        '''
        cursor = '*'
        stime = datetime.now()
        query_count = 0
        while True:
            #Get data with starting cursorMark
            query = self._get_query(cursor)
            #Add FQ to the query. This is used by resume to filter on date fields and when specifying document subset.
            #Not included in _get_query for more flexibiilty.

            if fq:
                if 'fq' in query:
                    [query['fq'].append(x) for x in fq]
                else:
                    query['fq'] = fq

            results = self._source.query(self._source_coll, query)
            query_count += 1
            if query_count % report_frequency == 0:
                self.log.info("Processed {} Items in {} Seconds. Apprximately {} items/minute".format(
                            self._items_processed, int((datetime.now()-stime).seconds),
                            str(int(self._items_processed / ((datetime.now()-stime).seconds/60)))
                            ))

            if results.get_results_count():
                #If we got items back, get the new cursor and yield the docs
                self._items_processed += results.get_results_count()
                cursor = results.get_cursor()
                #Remove ignore fields
                docs = self._trim_fields(results.docs)
                yield docs
                if results.get_results_count() < self._rows:
                    #Less results than asked, probably done
                    break
            else:
                #No Results, probably done :)
                self.log.debug("Got zero Results with cursor: {}".format(cursor))
                break