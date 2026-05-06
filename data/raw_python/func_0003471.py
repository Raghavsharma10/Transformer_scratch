def _connect_deferred(self, deferred):
        """
        Hook up the Deferred that that this will be the result of.

        Should only be run in Twisted thread, and only called once.
        """
        self._deferred = deferred

        # Because we use __del__, we need to make sure there are no cycles
        # involving this object, which is why we use a weakref:
        def put(result, eventual=weakref.ref(self)):
            eventual = eventual()
            if eventual:
                eventual._set_result(result)
            else:
                err(result, "Unhandled error in EventualResult")

        deferred.addBoth(put)