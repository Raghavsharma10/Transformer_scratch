def publish_queue(self):
        """
        Publish all messages that have been added to the queue for configured protocol
        :return: None
        """
        self.last_send_time = time.time()
        try:
            self._tx_queue_lock.acquire()
            start_length = len(self._rx_queue)
            publish_amount = len(self._tx_queue)
            if self.config.protocol == PublisherConfig.Protocol.GRPC:
                self._publish_queue_grpc()
            else:
                self._publish_queue_wss()
            self._tx_queue = []
        finally:
            self._tx_queue_lock.release()

        if self.config.publish_type == self.config.Type.SYNC:
            start_time = time.time()
            while time.time() - start_time < self.config.sync_timeout and \
                                    len(self._rx_queue) - start_length < publish_amount:
                pass
            return self._rx_queue