def register(self, model, related=None):
        '''Register a :class:`StdModel` with this search :class:`SearchEngine`.
When registering a model, every time an instance is created, it will be
indexed by the search engine.

:param model: a :class:`StdModel` class.
:param related: a list of related fields to include in the index.
'''
        update_model = UpdateSE(self, related)
        self.REGISTERED_MODELS[model] = update_model
        self.router.post_commit.bind(update_model, model)
        self.router.post_delete.bind(update_model, model)