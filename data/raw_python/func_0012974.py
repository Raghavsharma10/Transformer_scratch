def map(self, callback, pass_batch_into_callback=None,
          merge_future=None, **q_options):
    """Map a callback function or tasklet over the query results.

    Args:
      callback: A function or tasklet to be applied to each result; see below.
      merge_future: Optional Future subclass; see below.
      **q_options: All query options keyword arguments are supported.

    Callback signature: The callback is normally called with an entity
    as argument.  However if keys_only=True is given, it is called
    with a Key.  Also, when pass_batch_into_callback is True, it is
    called with three arguments: the current batch, the index within
    the batch, and the entity or Key at that index.  The callback can
    return whatever it wants.  If the callback is None, a trivial
    callback is assumed that just returns the entity or key passed in
    (ignoring produce_cursors).

    Optional merge future: The merge_future is an advanced argument
    that can be used to override how the callback results are combined
    into the overall map() return value.  By default a list of
    callback return values is produced.  By substituting one of a
    small number of specialized alternatives you can arrange
    otherwise.  See tasklets.MultiFuture for the default
    implementation and a description of the protocol the merge_future
    object must implement the default.  Alternatives from the same
    module include QueueFuture, SerialQueueFuture and ReducingFuture.

    Returns:
      When the query has run to completion and all callbacks have
      returned, map() returns a list of the results of all callbacks.
      (But see 'optional merge future' above.)
    """
    return self.map_async(callback,
                          pass_batch_into_callback=pass_batch_into_callback,
                          merge_future=merge_future,
                          **q_options).get_result()