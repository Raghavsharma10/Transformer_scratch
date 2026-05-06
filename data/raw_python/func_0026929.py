def timesince(self, when):
        """
        Returns human friendly version of the timespan between now
        and the given datetime.
        """
        units = (
            ("year",   60 * 60 * 24 * 365),
            ("week",   60 * 60 * 24 * 7),
            ("day",    60 * 60 * 24),
            ("hour",   60 * 60),
            ("minute", 60),
            ("second", 1),
        )
        delta = datetime.now() - when
        total_seconds = delta.days * 60 * 60 * 24 + delta.seconds
        parts = []
        for name, seconds in units:
            value = total_seconds / seconds
            if value > 0:
                total_seconds %= seconds
                s = "s" if value != 1 else ""
                parts.append("%s %s%s" % (value, name, s))
        return " and ".join(", ".join(parts).rsplit(", ", 1))