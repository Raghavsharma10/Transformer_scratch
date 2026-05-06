def send_acks(self, message):
        """
        send acks to the service
        :param message: EventHub_pb2.Message
        :return: None
        """
        if isinstance(message, EventHub_pb2.Message):
            ack = EventHub_pb2.Ack(partition=message.partition, offset=message.offset)
            self.grpc_manager.send_message(EventHub_pb2.SubscriptionResponse(ack=ack))

        elif isinstance(message, EventHub_pb2.SubscriptionMessage):
            acks = []
            for m in message.messages:
                acks.append(EventHub_pb2.Ack(parition=m.partition, offset=m.offset))
            self.grpc_manager.send_message(EventHub_pb2.SubscriptionAcks(ack=acks))