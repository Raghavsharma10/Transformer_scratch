def fromseconds(cls, seconds):
        """Return a |Period| instance based on a given number of seconds."""
        try:
            seconds = int(seconds)
        except TypeError:
            seconds = int(seconds.flatten()[0])
        return cls(datetime.timedelta(0, int(seconds)))