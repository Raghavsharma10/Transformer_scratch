def _allocate_ids_async(cls, size=None, max=None, parent=None,
                          **ctx_options):
    """Allocates a range of key IDs for this model class.

    This is the asynchronous version of Model._allocate_ids().
    """
    from . import tasklets
    ctx = tasklets.get_context()
    cls._pre_allocate_ids_hook(size, max, parent)
    key = Key(cls._get_kind(), None, parent=parent)
    fut = ctx.allocate_ids(key, size=size, max=max, **ctx_options)
    post_hook = cls._post_allocate_ids_hook
    if not cls._is_default_hook(Model._default_post_allocate_ids_hook,
                                post_hook):
      fut.add_immediate_callback(post_hook, size, max, parent, fut)
    return fut