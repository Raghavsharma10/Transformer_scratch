def has_resolved_dependencies(self):
        """Return True if all dependencies are in State.DONE"""
        for dependency in self.dependencies:
            if dependency.state != Task.State.DONE:
                return False

        return True