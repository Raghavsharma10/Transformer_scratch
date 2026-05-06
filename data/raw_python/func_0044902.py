def hms2frame(hms, fps):
        """
        :param hms: a string, e.g. "01:23:15" for one hour, 23 minutes 15 seconds 
        :param fps: framerate 
        :return: frame number
        """
        import time
        t = time.strptime(hms, "%H:%M:%S")
        return (t.tm_hour * 60 * 60 + t.tm_min * 60 + t.tm_sec) * fps