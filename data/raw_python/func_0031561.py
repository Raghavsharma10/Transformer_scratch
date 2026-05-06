def randomUUIDField(self):
        """
        Return the unique uuid from uuid1, uuid3, uuid4, or uuid5.
        """
        uuid1 = uuid.uuid1().hex
        uuid3 = uuid.uuid3(
            uuid.NAMESPACE_URL,
            self.randomize(['python', 'django', 'awesome'])
        ).hex
        uuid4 = uuid.uuid4().hex
        uuid5 = uuid.uuid5(
            uuid.NAMESPACE_DNS,
            self.randomize(['python', 'django', 'awesome'])
        ).hex
        return self.randomize([uuid1, uuid3, uuid4, uuid5])