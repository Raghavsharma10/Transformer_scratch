def then(self, success=None, failure=None):
        """A utility method to add success and/or failure callback to the
        promise which will also return another promise in the process.
        """
        rv = Promise()

        def on_success(v):
            try:
                rv.resolve(success(v))
            except Exception as e:
                rv.reject(e)

        def on_failure(r):
            try:
                rv.resolve(failure(r))
            except Exception as e:
                rv.reject(e)

        self.done(on_success, on_failure)
        return rv