def send(self, topic, *args, **kwargs):
        """
        Appends the prefix to the topic before sendingf
        """
        prefix_topic = self.heroku_kafka.prefix_topic(topic)
        return super(HerokuKafkaProducer, self).send(prefix_topic, *args, **kwargs)