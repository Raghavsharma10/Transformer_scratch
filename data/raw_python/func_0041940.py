def token(cls: Type[CSVType], time: int) -> CSVType:
        """
        Return CSV instance from time

        :param time: Timestamp
        :return:
        """
        csv = cls()
        csv.time = str(time)
        return csv