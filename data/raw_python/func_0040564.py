def create(cls, description="", message=""):
        """
        :param description:
        :type description: str
        :param message:
        :type message: str
        """
        instance = cls()
        instance.revision_id = make_hash_id()
        instance.release_date = datetime.datetime.now()

        if len(description) > 0:
            instance.description = description

        if len(message) > 0:
            instance.message = message

        return instance