def _find_snapshot(self, name):
        """
        Find snapshot on remote by name or regular expression
        """
        remote_snapshots = self._get_snapshots(self.client)
        for remote in reversed(remote_snapshots):
            if remote["Name"] == name or \
                    re.match(name, remote["Name"]):
                return remote
        return None