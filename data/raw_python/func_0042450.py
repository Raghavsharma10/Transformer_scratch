def send(self, topic, message):
        """Publishes a pulse message to the proper exchange."""

        if not message:
            Log.error("Expecting a message")

        message._prepare()

        if not self.connection:
            self.connect()

        producer = Producer(
            channel=self.connection,
            exchange=Exchange(self.settings.exchange, type='topic'),
            routing_key=topic
        )

        # The message is actually a simple envelope format with a payload and
        # some metadata.
        final_data = Data(
            payload=message.data,
            _meta=set_default({
                'exchange': self.settings.exchange,
                'routing_key': message.routing_key,
                'serializer': self.settings.serializer,
                'sent': time_to_string(datetime.datetime.now(timezone(self.settings.broker_timezone))),
                'count': self.count
            }, message.metadata)
        )

        producer.publish(jsons.scrub(final_data), serializer=self.settings.serializer)
        self.count += 1