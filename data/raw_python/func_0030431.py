def asStructTime(self, tzinfo=None):
        """Return this time represented as a time.struct_time.

        tzinfo is a datetime.tzinfo instance coresponding to the desired
        timezone of the output. If is is the default None, UTC is assumed.
        """
        dtime = self.asDatetime(tzinfo)
        if tzinfo is None:
            return dtime.utctimetuple()
        else:
            return dtime.timetuple()