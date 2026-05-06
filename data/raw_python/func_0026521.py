def rec(self):
        """Records a single snapshot"""

        try:
            self._snapshot()
        except Exception as e:
            self.log("Timer error: ", e, type(e), lvl=error)