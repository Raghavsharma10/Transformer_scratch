def from_time(cls, other):
        """Construct an :class:`nptime` object from a :class:`datetime.time`

        .. note::
            This *ignores* the :class:`datetime.tzinfo` that may be part
            of the ``time`` object.
        """
        return cls(other.hour, other.minute, other.second, other.microsecond)