def build_duration(self):
        """Return the difference between build and build_done states"""

        return int(self.state.build_done) - int(self.state.build)