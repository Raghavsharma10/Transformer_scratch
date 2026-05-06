def compare(self, other, components=[]):
        """
        Compare two publishes
        It expects that other publish is same or older than this one

        Return tuple (diff, equal) of dict {'component': ['snapshot']}
        """
        lg.debug("Comparing publish %s (%s) and %s (%s)" % (self.name, self.storage or "local", other.name, other.storage or "local"))

        diff, equal = ({}, {})

        for component, snapshots in self.components.items():
            if component not in list(other.components.keys()):
                # Component is missing in other
                diff[component] = snapshots
                continue

            equal_snapshots = list(set(snapshots).intersection(other.components[component]))
            if equal_snapshots:
                lg.debug("Equal snapshots for %s: %s" % (component, equal_snapshots))
                equal[component] = equal_snapshots

            diff_snapshots = list(set(snapshots).difference(other.components[component]))
            if diff_snapshots:
                lg.debug("Different snapshots for %s: %s" % (component, diff_snapshots))
                diff[component] = diff_snapshots

        return (diff, equal)