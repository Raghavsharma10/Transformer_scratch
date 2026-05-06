def begin(self, **options):
        '''Begin a new :class:`Transaction`. If this :class:`Session`
is already in a :ref:`transactional state <transactional-state>`,
an error will occur. It returns the :attr:`transaction` attribute.

This method is mostly used within a ``with`` statement block::

    with session.begin() as t:
        t.add(...)
        ...

which is equivalent to::

    t = session.begin()
    t.add(...)
    ...
    session.commit()

``options`` parameters are passed to the :class:`Transaction` constructor.
'''
        if self.transaction is not None:
            raise InvalidTransaction("A transaction is already begun.")
        else:
            self.transaction = Transaction(self, **options)
        return self.transaction