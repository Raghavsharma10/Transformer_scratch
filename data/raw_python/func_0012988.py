def run_to_queue(self, queue, conn, options=None):
    """Run this query, putting entities into the given queue."""
    if options is None:
      # Default options.
      offset = None
      limit = None
      keys_only = None
    else:
      # Capture options we need to simulate.
      offset = options.offset
      limit = options.limit
      keys_only = options.keys_only

      # Cursors are supported for certain orders only.
      if (options.start_cursor or options.end_cursor or
          options.produce_cursors):
        names = set()
        if self.__orders is not None:
          names = self.__orders._get_prop_names()
        if '__key__' not in names:
          raise datastore_errors.BadArgumentError(
              '_MultiQuery with cursors requires __key__ order')

    # Decide if we need to modify the options passed to subqueries.
    # NOTE: It would seem we can sometimes let Cloud Datastore handle
    # the offset natively, but this would thwart the duplicate key
    # detection, so we always have to emulate the offset here.
    # We can set the limit we pass along to offset + limit though,
    # since that is the maximum number of results from a single
    # subquery we will ever have to consider.
    modifiers = {}
    if offset:
      modifiers['offset'] = None
      if limit is not None:
        modifiers['limit'] = min(_MAX_LIMIT, offset + limit)
    if keys_only and self.__orders is not None:
      modifiers['keys_only'] = None
    if modifiers:
      options = QueryOptions(config=options, **modifiers)

    if offset is None:
      offset = 0

    if limit is None:
      limit = _MAX_LIMIT

    if self.__orders is None:
      # Run the subqueries sequentially; there is no order to keep.
      keys_seen = set()
      for subq in self.__subqueries:
        if limit <= 0:
          break
        subit = tasklets.SerialQueueFuture('_MultiQuery.run_to_queue[ser]')
        subq.run_to_queue(subit, conn, options=options)
        while limit > 0:
          try:
            batch, index, result = yield subit.getq()
          except EOFError:
            break
          if keys_only:
            key = result
          else:
            key = result._key
          if key not in keys_seen:
            keys_seen.add(key)
            if offset > 0:
              offset -= 1
            else:
              limit -= 1
              queue.putq((None, None, result))
      queue.complete()
      return

    # This with-statement causes the adapter to set _orig_pb on all
    # entities it converts from protobuf.
    # TODO: Does this interact properly with the cache?
    with conn.adapter:
      # Start running all the sub-queries.
      todo = []  # List of (subit, dsquery) tuples.
      for subq in self.__subqueries:
        dsquery = subq._get_query(conn)
        subit = tasklets.SerialQueueFuture('_MultiQuery.run_to_queue[par]')
        subq.run_to_queue(subit, conn, options=options, dsquery=dsquery)
        todo.append((subit, dsquery))

      # Create a list of (first-entity, subquery-iterator) tuples.
      state = []  # List of _SubQueryIteratorState instances.
      for subit, dsquery in todo:
        try:
          thing = yield subit.getq()
        except EOFError:
          continue
        else:
          state.append(_SubQueryIteratorState(thing, subit, dsquery,
                                              self.__orders))

      # Now turn it into a sorted heap.  The heapq module claims that
      # calling heapify() is more efficient than calling heappush() for
      # each item.
      heapq.heapify(state)

      # Repeatedly yield the lowest entity from the state vector,
      # filtering duplicates.  This is essentially a multi-way merge
      # sort.  One would think it should be possible to filter
      # duplicates simply by dropping other entities already in the
      # state vector that are equal to the lowest entity, but because of
      # the weird sorting of repeated properties, we have to explicitly
      # keep a set of all keys, so we can remove later occurrences.
      # Note that entities will still be sorted correctly, within the
      # constraints given by the sort order.
      keys_seen = set()
      while state and limit > 0:
        item = heapq.heappop(state)
        batch = item.batch
        index = item.index
        entity = item.entity
        key = entity._key
        if key not in keys_seen:
          keys_seen.add(key)
          if offset > 0:
            offset -= 1
          else:
            limit -= 1
            if keys_only:
              queue.putq((batch, index, key))
            else:
              queue.putq((batch, index, entity))
        subit = item.iterator
        try:
          batch, index, entity = yield subit.getq()
        except EOFError:
          pass
        else:
          item.batch = batch
          item.index = index
          item.entity = entity
          heapq.heappush(state, item)
      queue.complete()