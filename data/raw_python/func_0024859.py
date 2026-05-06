def _publish_queue_grpc(self):
        """
        send the messages in the tx queue to the GRPC manager
        :return: None
        """
        messages = EventHub_pb2.Messages(msg=self._tx_queue)
        publish_request = EventHub_pb2.PublishRequest(messages=messages)
        self.grpc_manager.send_message(publish_request)