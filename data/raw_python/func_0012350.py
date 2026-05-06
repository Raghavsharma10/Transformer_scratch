def load(self):
        """
        Load publish info from remote
        """
        publish = self._get_publish()
        self.architectures = publish['Architectures']
        for source in publish['Sources']:
            component = source['Component']
            snapshot = source['Name']
            self.publish_snapshots.append({
                'Component': component,
                'Name': snapshot
            })

            snapshot_remote = self._find_snapshot(snapshot)
            for source in self._get_source_snapshots(snapshot_remote, fallback_self=True):
                self.add(source, component)