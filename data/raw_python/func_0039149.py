def output_package(dist):
        """Return string displaying package information."""
        if dist_is_editable(dist):
            return '%s (%s, %s)' % (
                dist.project_name,
                dist.version,
                dist.location,
            )
        return '%s (%s)' % (dist.project_name, dist.version)