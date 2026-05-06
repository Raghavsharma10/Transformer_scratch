def backfill_history(self, num_days, available_table_names):
        """
        Backfill historical data for days that are missing.

        :param num_days: number of days of historical data to backfill,
          if missing
        :type num_days: int
        :param available_table_names: names of available per-date tables
        :type available_table_names: ``list``
        """
        if num_days == -1:
            # skip the first date, under the assumption that data may be
            # incomplete
            logger.info('Backfilling all available history')
            start_table = available_table_names[1]
        else:
            logger.info('Backfilling %d days of history', num_days)
            start_table = available_table_names[-1 * num_days]
        start_date = self._datetime_for_table_name(start_table)
        end_table = available_table_names[-3]
        end_date = self._datetime_for_table_name(end_table)
        logger.debug(
            'Backfilling history from %s (%s) to %s (%s)', start_table,
            start_date.strftime('%Y-%m-%d'), end_table,
            end_date.strftime('%Y-%m-%d')
        )
        for days in range((end_date - start_date).days + 1):
            backfill_dt = start_date + timedelta(days=days)
            if self._have_cache_for_date(backfill_dt):
                logger.info('Cache present for all projects for %s; skipping',
                            backfill_dt.strftime('%Y-%m-%d'))
                continue
            backfill_table = self._table_name_for_datetime(backfill_dt)
            logger.info('Backfilling %s (%s)', backfill_table,
                        backfill_dt.strftime('%Y-%m-%d'))
            self.query_one_table(backfill_table)