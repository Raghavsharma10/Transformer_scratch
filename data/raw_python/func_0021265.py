def _convert_value_to_native(value):
    """
    Converts pysnmp objects into native Python objects.
    """
    if isinstance(value, Counter32):
        return int(value.prettyPrint())
    if isinstance(value, Counter64):
        return int(value.prettyPrint())
    if isinstance(value, Gauge32):
        return int(value.prettyPrint())
    if isinstance(value, Integer):
        return int(value.prettyPrint())
    if isinstance(value, Integer32):
        return int(value.prettyPrint())
    if isinstance(value, Unsigned32):
        return int(value.prettyPrint())
    if isinstance(value, IpAddress):
        return str(value.prettyPrint())
    if isinstance(value, OctetString):
        try:
            return value.asOctets().decode(value.encoding)
        except UnicodeDecodeError:
            return value.asOctets()
    if isinstance(value, TimeTicks):
        return timedelta(seconds=int(value.prettyPrint()) / 100.0)
    return value