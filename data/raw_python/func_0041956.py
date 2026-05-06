def transaction(self, callback):
        """Executes a function in a transaction.

        The function gets passed this Connection instance as an (optional) parameter.

        If an exception occurs during execution of the function or transaction commit,
        the transaction is rolled back and the exception re-thrown.

        :param callback: the function to execute in a transaction
        :return: the value returned by the `callback`
        :raise: Exception
        """
        self.begin_transaction()
        try:
            result = callback(self)
            self.commit()
            return result
        except:
            self.rollback()
            raise