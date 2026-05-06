def reconnect(self, conn_retries=None):
        ''' Connects to Redis with a exponential waiting (3**n) '''
        if conn_retries is None:
            conn_retries = self.conn_retries

        count = 0
        if self.logger:
            self.logger.info('Connecting to Redis..')
        while count < conn_retries:
            super(redis.client.Redis, self).__init__(*self.args, **self.kwargs)

            if self.ping():
                if self.logger:
                    self.logger.info('Connected to Redis!')
                return True
            else:
                sl = min(3 ** count, self.max_sleep)
                if self.logger:
                    self.logger.info('Connecting failed, retrying in {0} seconds'.format(sl))
                time.sleep(sl)
                count += 1
        raise ConnectionError