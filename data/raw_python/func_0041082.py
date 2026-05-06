def __load_dump(self, message):
        """
        Calls the hook method to modify the loaded peer description before
        giving it to the directory

        :param message: The received Herald message
        :return: The updated peer description
        """
        dump = message.content
        if self._hook is not None:
            # Call the hook
            try:
                updated_dump = self._hook(message, dump)
                if updated_dump is not None:
                    # Use the new description
                    dump = updated_dump
            except (TypeError, ValueError) as ex:
                self._logger("Invalid description hook: %s", ex)
        return dump