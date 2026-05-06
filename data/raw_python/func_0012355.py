def _get_source_snapshots(self, snapshot, fallback_self=False):
        """
        Get list of source snapshot names of given snapshot

        TODO: we have to decide by description at the moment
        """
        if not snapshot:
            return []

        source_snapshots = re.findall(r"'([\w\d\.-]+)'", snapshot['Description'])
        if not source_snapshots and fallback_self:
            source_snapshots = [snapshot['Name']]

        source_snapshots.sort()
        return source_snapshots