def __reflect(self):
        """Reflect metadata
        """

        def only(name, _):
            return self.__only(name) and self.__mapper.restore_bucket(name) is not None

        self.__metadata.reflect(only=only)