def _publisher_callback(self, publish_ack):
        """
        publisher callback that grpc and web socket can pass messages to
        address the received message onto the queue
        :param publish_ack: EventHub_pb2.Ack the ack received from either wss or grpc
        :return: None
        """
        logging.debug("ack received: " + str(publish_ack).replace('\n', ' '))
        self._rx_queue.append(publish_ack)