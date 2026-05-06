def reject(self, *, requeue=True):
        """
        Reject the message.

        :keyword bool requeue: if true, the broker will attempt to requeue the
            message and deliver it to an alternate consumer.
        """
        self.sender.send_BasicReject(self.delivery_tag, requeue)