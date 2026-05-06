def run_queries(self, backfill_num_days=7):
        """
        Run the data queries for the specified projects.

        :param backfill_num_days: number of days of historical data to backfill,
          if missing
        :type backfill_num_days: int
        """
        available_tables = self._get_download_table_ids()
        logger.debug('Found %d available download tables: %s',
                     len(available_tables), available_tables)
        today_table = available_tables[-1]
        yesterday_table = available_tables[-2]
        self.query_one_table(today_table)
        self.query_one_table(yesterday_table)
        self.backfill_history(backfill_num_days, available_tables)