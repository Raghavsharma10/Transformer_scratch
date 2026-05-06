def basekey(self, meta, *args):
        """Calculate the key to access model data.

:parameter meta: a :class:`stdnet.odm.Metaclass`.
:parameter args: optional list of strings to prepend to the basekey.
:rtype: a native string
"""
        key = '%s%s' % (self.namespace, meta.modelkey)
        postfix = ':'.join((str(p) for p in args if p is not None))
        return '%s:%s' % (key, postfix) if postfix else key