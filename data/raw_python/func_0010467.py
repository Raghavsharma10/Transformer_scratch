def _process_message(self, message: amqp.Message) -> None:
        """Processes the message received from the queue."""
        if self.shutdown_pending.is_set():
            return

        try:
            if isinstance(message.body, bytes):
                message.body = message.body.decode()
            description = json.loads(message.body)
        except Exception:
            logger.error("Cannot decode message. Dropping. Message: %r", message.body)
            traceback.print_exc()
            message.channel.basic_reject(message.delivery_tag, requeue=False)
        else:
            logger.info("Processing task: %r", description)
            self._process_description(message, description)