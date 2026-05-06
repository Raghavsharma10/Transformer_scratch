def topic_inject(self, topic_name, _msg_content=None, **kwargs):
        """
        Injecting message into topic. if _msg_content, we inject it directly. if not, we use all extra kwargs
        :param topic_name: name of the topic
        :param _msg_content: optional message content
        :param kwargs: each extra kwarg will be put int he message is structure matches
        :return:
        """
        #changing unicode to string ( testing stability of multiprocess debugging )
        if isinstance(topic_name, unicode):
            topic_name = unicodedata.normalize('NFKD', topic_name).encode('ascii', 'ignore')

        if _msg_content is not None:
            # logging.warn("injecting {msg} into {topic}".format(msg=_msg_content, topic=topic_name))
            res = self.topic_svc.call(args=(topic_name, _msg_content,))
        else:  # default kwargs is {}
            # logging.warn("injecting {msg} into {topic}".format(msg=kwargs, topic=topic_name))
            res = self.topic_svc.call(args=(topic_name, kwargs,))

        return res is None