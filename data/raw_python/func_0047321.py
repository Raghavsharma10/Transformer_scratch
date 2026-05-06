def build_items(self):
        """
        get the items from STATS QUEUE
        calculate self.stats
        make new items from self.stats
        put the new items for ITEM QUEUE
        """
        while not self.stats_queue.empty():
            item = self.stats_queue.get()
            self.calculate(item)

        for key, value in self.stats.iteritems():
            if 'blackbird.queue.length' == key:
                value = self.queue.qsize()
            item = BlackbirdStatisticsItem(
                key=key,
                value=value,
                host=self.options['hostname']
            )
            if self.enqueue(item=item, queue=self.queue):
                self.logger.debug(
                    'Inserted {0} to the queue.'.format(item.data)
                )