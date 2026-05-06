def loads(content, dict_=dict):
    """Parse a toml string
    An additional argument `dict_` is used to specify the output type
    """
    if not isinstance(content, basestring):
        raise ValueError('The first parameter needs to be a string object, ',
                         '%r is passed' % type(content))
    decoder = Decoder(content, dict_)
    decoder.parse()
    return decoder.data