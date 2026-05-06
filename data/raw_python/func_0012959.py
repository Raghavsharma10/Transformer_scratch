def queue_rpc(self, rpc, callback=None, *args, **kwds):
    """Schedule an RPC with an optional callback.

    The caller must have previously sent the call to the service.
    The optional callback is called with the remaining arguments.

    NOTE: If the rpc is a MultiRpc, the callback will be called once
    for each sub-RPC.  TODO: Is this a good idea?
    """
    if rpc is None:
      return
    if rpc.state not in (_RUNNING, _FINISHING):
      raise RuntimeError('rpc must be sent to service before queueing')
    if isinstance(rpc, datastore_rpc.MultiRpc):
      rpcs = rpc.rpcs
      if len(rpcs) > 1:
        # Don't call the callback until all sub-rpcs have completed.
        rpc.__done = False

        def help_multi_rpc_along(r=rpc, c=callback, a=args, k=kwds):
          if r.state == _FINISHING and not r.__done:
            r.__done = True
            c(*a, **k)
            # TODO: And again, what about exceptions?
        callback = help_multi_rpc_along
        args = ()
        kwds = {}
    else:
      rpcs = [rpc]
    for rpc in rpcs:
      self.rpcs[rpc] = (callback, args, kwds)