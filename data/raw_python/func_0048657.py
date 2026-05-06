def post_dispatch(self):
        """
        List of post-dispatch methods from registered middleware.
        """
        middleware = sort_by_priority(self, reverse=True)
        return tuple(m.post_dispatch for m in middleware if hasattr(m, 'post_dispatch'))