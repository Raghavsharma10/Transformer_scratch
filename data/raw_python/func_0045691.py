def main(self):
        """Search and sieve query names."""
        # TODO: Break up, too complex
        primary_bool = True
        no_records = True
        nsearch = 1
        search_terms = self.terms
        original_names = []
        while True:
            if primary_bool:
                self.logger.info('Searching [{0}] ...'.format(
                    self.primary_datasource))
            else:
                self.logger.info('Searching other datasources ...')
            res = self._res.search(search_terms, prelim=primary_bool)
            if nsearch > 2 and res:
                # if second search failed, look up alternative names
                for each_res, original_name in zip(res, original_names):
                    each_res['supplied_name_string'] = original_name
            self._store.add(res)
            # Check for returns without records
            no_records = self._count(nrecords=1)
            if no_records:
                if nsearch == 1:
                    primary_bool = False
                elif nsearch == 2:
                    original_names = no_records
                    # genus names
                    no_records = [e.split()[0] for e in no_records]
                    primary_bool = True
                elif nsearch == 3:
                    original_names = no_records
                    no_records = [e.split()[0] for e in no_records]
                    primary_bool = False
                else:
                    break
            else:
                break
            nsearch += 1
            search_terms = no_records
        # Check for multiple records
        multi_records = self._count(greater=True, nrecords=1)
        if multi_records:
            self.logger.info('Choosing best records to return ...')
            res = self._sieve(multi_records)
            self._store.replace(res)