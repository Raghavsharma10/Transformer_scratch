def _map(self, data_item):
        "Map ``data_item`` separately in each thread."
        delegate = self.delegate
        logger.debug(f'mapping: {data_item}')
        if self.clobber or not self.exists(data_item.id):
            logger.debug(f'exist: {data_item.id}: {self.exists(data_item.id)}')
            delegate.dump(data_item.id, data_item)