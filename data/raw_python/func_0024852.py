def add_message(self, id, body, tags=False):
        """
        add messages to the rx_queue
        :param id: str message Id
        :param body: str the message body
        :param tags: dict[string->string] tags to be associated with the message
        :return: self
        """
        if not tags:
            tags = {}
        try:
            self._tx_queue_lock.acquire()
            self._tx_queue.append(
                EventHub_pb2.Message(id=id, body=body, tags=tags, zone_id=self.eventhub_client.zone_id))
        finally:
            self._tx_queue_lock.release()
        return self