def _type_single(self, value, _type):
        ' apply type to the single value '
        if value is None or _type in (None, NoneType):
            # don't convert null values
            # default type is the original type if none set
            pass
        elif isinstance(value, _type):  # or values already of correct type
            # normalize all dates to epochs
            value = dt2ts(value) if _type in [datetime, date] else value
        else:
            if _type in (datetime, date):
                # normalize all dates to epochs
                value = dt2ts(value)
            elif _type in (unicode, str):
                # make sure all string types are properly unicoded
                value = to_encoding(value)
            else:
                try:
                    value = _type(value)
                except Exception:
                    value = to_encoding(value)
                    logger.error("typecast failed: %s(value=%s)" % (
                        _type.__name__, value))
                    raise
        return value