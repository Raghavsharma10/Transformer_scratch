def delete_async(self, **ctx_options):
    """Schedule deletion of the entity for this Key.

    This returns a Future, whose result becomes available once the
    deletion is complete.  If no such entity exists, a Future is still
    returned.  In all cases the Future's result is None (i.e. there is
    no way to tell whether the entity existed or not).
    """
    from . import tasklets, model
    ctx = tasklets.get_context()
    cls = model.Model._kind_map.get(self.kind())
    if cls:
      cls._pre_delete_hook(self)
    fut = ctx.delete(self, **ctx_options)
    if cls:
      post_hook = cls._post_delete_hook
      if not cls._is_default_hook(model.Model._default_post_delete_hook,
                                  post_hook):
        fut.add_immediate_callback(post_hook, self, fut)
    return fut