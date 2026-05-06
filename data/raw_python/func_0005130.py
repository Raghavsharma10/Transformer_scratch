def preprX(*attributes, address=True, full_name=False,
           pretty=False, keyless=False, **kwargs):
    """ `Creates prettier object representations`

        @*attributes: (#str) instance attributes within the object you
            wish to display. Attributes can be recursive
            e.g. |one.two.three| for access to |self.one.two.three|
        @address: (#bool) |True| to include the memory address
        @full_name: (#bool) |True| to include the full path to the
            object vs. the qualified name
        @pretty: (#bool) |True| to allow bolding and coloring
        @keyless: (#bool) |True| to display the values of @attributes
            withotu their attribute names
        ..
            class Foo(object):

                def __init__(self, bar, baz=None):
                    self.bar = bar
                    self.baz = baz

                __repr__ = prepr('bar', 'baz', address=False)

            foo = Foo('foobar')
            repr(foo)
        ..
        |<Foo:bar=`foobar`, baz=None>|
    """
    def _format(obj, attribute):
        try:
            if keyless:
                val = getattr_in(obj, attribute)
                if val is not None:
                    return repr(val)
            else:
                return '%s=%s' % (attribute,
                                  repr(getattr_in(obj, attribute)))
        except AttributeError:
            return None

    def prep(obj, address=address, full_name=full_name, pretty=pretty,
             keyless=keyless, **kwargs):
        if address:
            address = ":%s" % hex(id(obj))
        else:
            address = ""
        data = list(filter(lambda x: x is not None,
                           map(lambda a: _format(obj, a), attributes)))
        if data:
            data = ':%s' % ', '.join(data)
        else:
            data = ''
        return stdout_encode("<%s%s%s>" % (get_obj_name(obj), data, address))
    return prep