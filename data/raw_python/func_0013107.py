def get_async(self, **ctx_options):
    """Return a Future whose result is the entity for this Key.

    If no such entity exists, a Future is still returned, and the
    Future's eventual return result be None.
    """
    from . import model, tasklets
    ctx = tasklets.get_context()
    cls = model.Model._kind_map.get(self.kind())
    if cls:
      cls._pre_get_hook(self)
    fut = ctx.get(self, **ctx_options)
    if cls:
      post_hook = cls._post_get_hook
      if not cls._is_default_hook(model.Model._default_post_get_hook,
                                  post_hook):
        fut.add_immediate_callback(post_hook, self, fut)
    return fut