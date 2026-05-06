def check_if_exists(self):
        """Find an installed distribution that satisfies or conflicts
        with this requirement, and set self.satisfied_by or
        self.conflicts_with appropriately."""
        if self.req is None:
            return False
        try:
            self.satisfied_by = pkg_resources.get_distribution(self.req)
        except pkg_resources.DistributionNotFound:
            return False
        except pkg_resources.VersionConflict:
            self.conflicts_with = pkg_resources.get_distribution(self.req.project_name)
        return True