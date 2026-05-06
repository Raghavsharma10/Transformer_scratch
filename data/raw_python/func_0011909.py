def shutdown(self, signum, frame):  # pylint: disable=unused-argument
        """Shut it down"""
        if not self.exit:
            self.exit = True
            self.log.debug(f"SIGTRAP!{signum};{frame}")
            self.api.shutdown()
            self.strat.shutdown()