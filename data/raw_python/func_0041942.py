def token(cls: Type[CLTVType], timestamp: int) -> CLTVType:
        """
        Return CLTV instance from timestamp

        :param timestamp: Timestamp
        :return:
        """
        cltv = cls()
        cltv.timestamp = str(timestamp)
        return cltv