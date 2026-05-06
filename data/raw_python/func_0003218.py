def create_option_from_value(tag, value):
    """
    Set DHCP option with human friendly value
    """
    dhcp_option.parser()
    fake_opt = dhcp_option(tag = tag)
    for c in dhcp_option.subclasses:
        if c.criteria(fake_opt):
            if hasattr(c, '_parse_from_value'):
                return c(tag = tag, value = c._parse_from_value(value))
            else:
                raise ValueError('Invalid DHCP option ' + str(tag) + ": " + repr(value))
    else:
        fake_opt._setextra(_tobytes(value))
        return fake_opt