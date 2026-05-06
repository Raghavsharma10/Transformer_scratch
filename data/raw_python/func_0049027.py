def build_items(self):
        """
        main loop
        """

        conn = self.connect(address=self.server_address, port=self.server_port)

        if conn:

            while not self.queue.empty():
                item = self.queue.get()
                self.pool.append(item)
                if type(item.data) is (tuple or list):
                    self.body['data'].extend(item.data)
                else:
                    self.body['data'].append(item.data)
            self.logger.debug(self.body['data'])

            try:
                log_message = (
                    'Queue length is {0}'.format(len(self.body['data']))
                )
                self.logger.debug(log_message)
                if len(self.body['data']) != 0:
                    self.send(conn)
                    self.logger.debug(self.get_result())
            except:
                self._reverse_queue()
                log_message = (
                    'An error occurred.'
                    'Maybe socket error, or get invalid value.'
                )
                self.logger.debug(log_message)
            else:
                del self.body['data'][:]

        self.build_statistics_item()