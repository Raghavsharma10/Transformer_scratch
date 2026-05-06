def _get_or_insert_async(*args, **kwds):
    """Transactionally retrieves an existing entity or creates a new one.

    This is the asynchronous version of Model._get_or_insert().
    """
    # NOTE: The signature is really weird here because we want to support
    # models with properties named e.g. 'cls' or 'name'.
    from . import tasklets
    cls, name = args  # These must always be positional.
    get_arg = cls.__get_arg
    app = get_arg(kwds, 'app')
    namespace = get_arg(kwds, 'namespace')
    parent = get_arg(kwds, 'parent')
    context_options = get_arg(kwds, 'context_options')
    # (End of super-special argument parsing.)
    # TODO: Test the heck out of this, in all sorts of evil scenarios.
    if not isinstance(name, basestring):
      raise TypeError('name must be a string; received %r' % name)
    elif not name:
      raise ValueError('name cannot be an empty string.')
    key = Key(cls, name, app=app, namespace=namespace, parent=parent)

    @tasklets.tasklet
    def internal_tasklet():
      @tasklets.tasklet
      def txn():
        ent = yield key.get_async(options=context_options)
        if ent is None:
          ent = cls(**kwds)  # TODO: Use _populate().
          ent._key = key
          yield ent.put_async(options=context_options)
        raise tasklets.Return(ent)
      if in_transaction():
        # Run txn() in existing transaction.
        ent = yield txn()
      else:
        # Maybe avoid a transaction altogether.
        ent = yield key.get_async(options=context_options)
        if ent is None:
          # Run txn() in new transaction.
          ent = yield transaction_async(txn)
      raise tasklets.Return(ent)

    return internal_tasklet()