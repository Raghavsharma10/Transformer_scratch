def start_standby(cls, webdriver=None, max_time=WTF_TIMEOUT_MANAGER.EPIC, sleep=5):
        """
        Create an instance of BrowserStandBy() and immediately return a running instance.

        This is best used in a 'with' block.

        Example::

            with BrowserStandBy.start_standby():
                # Now browser is in standby, you can do a bunch of stuff with in this block.
                # ...

            # We are now outside the block, and the browser standby has ended.

        """
        return cls(webdriver=webdriver, max_time=max_time, sleep=sleep, _autostart=True)