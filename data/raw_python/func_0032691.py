def setex(self, name, value, time):
        """
        Set the value of key to ``value`` that expires in ``time``
        seconds. ``time`` can be represented by an integer or a Python
        timedelta object.

        :param name: str     the name of the redis key
        :param value: str
        :param time: secs
        :return: Future()
        """
        with self.pipe as pipe:
            return pipe.setex(self.redis_key(name),
                              value=self.valueparse.encode(value),
                              time=time)