def post_swagger(self):
        """
        List of post-swagger methods from registered middleware.

        This is used to modify documentation (eg add/remove any extra information, provided by the middleware)

        """
        middleware = sort_by_priority(self)
        return tuple(m.post_swagger for m in middleware if hasattr(m, 'post_swagger'))