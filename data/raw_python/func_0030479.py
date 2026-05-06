def __deserialize_datetime(self, string):
        """
        Deserializes string to datetime.

        The string should be in iso8601 datetime format.

        :param string: str.
        :return: datetime.
        """
        try:
            from dateutil.parser import parse
            timestr = str(datetime.fromtimestamp(string/1000))
            return parse(timestr)
        except ImportError:
            return string
        except ValueError:
            raise ApiException(
                status=0,
                reason=(
                    "Failed to parse `{0}` into a datetime object"
                    .format(string)
                )
            )