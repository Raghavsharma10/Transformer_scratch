def _freq_parser(self, freq):
        """Parse timedelta.

        Valid keywords "days", "day", "d", "hours", "hour", "h",
        "minutes", "minute", "min", "m", "seconds", "second", "sec", "s",
        "weeks", "week", "w",
        """
        freq = freq.lower().strip()

        valid_keywords = [
            "days", "day", "d",
            "hours", "hour", "h",
            "minutes", "minute", "min", "m",
            "seconds", "second", "sec", "s",
            "weeks", "week", "w",
        ]
        error_message = "'%s' is invalid, use one of %s" % (
            freq, valid_keywords)

        try:
            # day
            for surfix in ["days", "day", "d"]:
                if freq.endswith(surfix):
                    freq = freq.replace(surfix, "")
                    return timedelta(days=int(freq))

            # hour
            for surfix in ["hours", "hour", "h"]:
                if freq.endswith(surfix):
                    freq = freq.replace(surfix, "")
                    return timedelta(hours=int(freq))

            # minute
            for surfix in ["minutes", "minute", "min", "m"]:
                if freq.endswith(surfix):
                    freq = freq.replace(surfix, "")
                    return timedelta(minutes=int(freq))

            # second
            for surfix in ["seconds", "second", "sec", "s"]:
                if freq.endswith(surfix):
                    freq = freq.replace(surfix, "")
                    return timedelta(seconds=int(freq))

            # week
            for surfix in ["weeks", "week", "w"]:
                if freq.endswith(surfix):
                    freq = freq.replace(surfix, "")
                    return timedelta(days=int(freq) * 7)
        except:
            pass

        raise ValueError(error_message)