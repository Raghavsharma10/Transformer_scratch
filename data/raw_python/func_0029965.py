def set_last_access(self, tag):
        """Mark the time that this bundle was last accessed"""
        import time
        # time defeats check that value didn't change

        self.buildstate.access.last = '{}-{}'.format(tag, time.time())
        self.buildstate.commit()