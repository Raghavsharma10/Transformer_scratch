def get(self, key, **ctx_options):
    """Return a Model instance given the entity key.

    It will use the context cache if the cache policy for the given
    key is enabled.

    Args:
      key: Key instance.
      **ctx_options: Context options.

    Returns:
      A Model instance if the key exists in the datastore; None otherwise.
    """
    options = _make_ctx_options(ctx_options)
    use_cache = self._use_cache(key, options)
    if use_cache:
      self._load_from_cache_if_available(key)

    use_datastore = self._use_datastore(key, options)
    if (use_datastore and
        isinstance(self._conn, datastore_rpc.TransactionalConnection)):
      use_memcache = False
    else:
      use_memcache = self._use_memcache(key, options)
    ns = key.namespace()
    memcache_deadline = None  # Avoid worries about uninitialized variable.

    if use_memcache:
      mkey = self._memcache_prefix + key.urlsafe()
      memcache_deadline = self._get_memcache_deadline(options)
      mvalue = yield self.memcache_get(mkey, for_cas=use_datastore,
                                       namespace=ns, use_cache=True,
                                       deadline=memcache_deadline)
      # A value may have appeared while yielding.
      if use_cache:
        self._load_from_cache_if_available(key)
      if mvalue not in (_LOCKED, None):
        cls = model.Model._lookup_model(key.kind(),
                                        self._conn.adapter.default_model)
        pb = entity_pb.EntityProto()

        try:
          pb.MergePartialFromString(mvalue)
        except ProtocolBuffer.ProtocolBufferDecodeError:
          logging.warning('Corrupt memcache entry found '
                          'with key %s and namespace %s' % (mkey, ns))
          mvalue = None
        else:
          entity = cls._from_pb(pb)
          # Store the key on the entity since it wasn't written to memcache.
          entity._key = key
          if use_cache:
            # Update in-memory cache.
            self._cache[key] = entity
          raise tasklets.Return(entity)

      if mvalue is None and use_datastore:
        yield self.memcache_set(mkey, _LOCKED, time=_LOCK_TIME, namespace=ns,
                                use_cache=True, deadline=memcache_deadline)
        yield self.memcache_gets(mkey, namespace=ns, use_cache=True,
                                 deadline=memcache_deadline)

    if not use_datastore:
      # NOTE: Do not cache this miss.  In some scenarios this would
      # prevent an app from working properly.
      raise tasklets.Return(None)

    if use_cache:
      entity = yield self._get_batcher.add_once(key, options)
    else:
      entity = yield self._get_batcher.add(key, options)

    if entity is not None:
      if use_memcache and mvalue != _LOCKED:
        # Don't serialize the key since it's already the memcache key.
        pbs = entity._to_pb(set_key=False).SerializePartialToString()
        # Don't attempt to write to memcache if too big.  Note that we
        # use LBYL ("look before you leap") because a multi-value
        # memcache operation would fail for all entities rather than
        # for just the one that's too big.  (Also, the AutoBatcher
        # class doesn't pass back exceptions very well.)
        if len(pbs) <= memcache.MAX_VALUE_SIZE:
          timeout = self._get_memcache_timeout(key, options)
          # Don't use fire-and-forget -- for users who forget
          # @ndb.toplevel, it's too painful to diagnose why their simple
          # code using a single synchronous call doesn't seem to use
          # memcache.  See issue 105.  http://goo.gl/JQZxp
          yield self.memcache_cas(mkey, pbs, time=timeout, namespace=ns,
                                  deadline=memcache_deadline)

    if use_cache:
      # Cache hit or miss.  NOTE: In this case it is okay to cache a
      # miss; the datastore is the ultimate authority.
      self._cache[key] = entity

    raise tasklets.Return(entity)