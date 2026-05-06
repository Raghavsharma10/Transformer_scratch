async def wait_for_all_to_process(self, *matchers, eventlist = None, eventdict = None,
                                            callback = None):
        """
        Similar to `waitForAll`, but set `canignore=True` for these events. This ensures
        blocking events are processed correctly.
        """
        def _callback(event, matcher):
            event.canignore = True
            if callback:
                callback(event, matcher)
        return await self.wait_for_all(*matchers, eventlist=eventlist,
                                       eventdict=eventdict, callback=_callback)