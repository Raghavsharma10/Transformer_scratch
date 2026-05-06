def _put_async(self, **ctx_options):
    """Write this entity to Cloud Datastore.

    This is the asynchronous version of Model._put().
    """
    if self._projection:
      raise datastore_errors.BadRequestError('Cannot put a partial entity')
    from . import tasklets
    ctx = tasklets.get_context()
    self._prepare_for_put()
    if self._key is None:
      self._key = Key(self._get_kind(), None)
    self._pre_put_hook()
    fut = ctx.put(self, **ctx_options)
    post_hook = self._post_put_hook
    if not self._is_default_hook(Model._default_post_put_hook, post_hook):
      fut.add_immediate_callback(post_hook, fut)
    return fut