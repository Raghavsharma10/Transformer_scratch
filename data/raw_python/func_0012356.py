def merge_snapshots(self):
        """
        Create component snapshots by merging other snapshots of same component
        """
        self.publish_snapshots = []
        for component, snapshots in self.components.items():
            if len(snapshots) <= 1:
                # Only one snapshot, no need to merge
                lg.debug("Component %s has only one snapshot %s, not creating merge snapshot" % (component, snapshots))
                self.publish_snapshots.append({
                    'Component': component,
                    'Name': snapshots[0]
                })
                continue

            # Look if merged snapshot doesn't already exist
            remote_snapshot = self._find_snapshot(r'^%s%s-%s-\d+' % (self.merge_prefix, self.name.replace('./', '').replace('/', '-'), component))
            if remote_snapshot:
                source_snapshots = self._get_source_snapshots(remote_snapshot)

                # Check if latest merged snapshot has same source snapshots like us
                snapshots_want = list(snapshots)
                snapshots_want.sort()

                lg.debug("Comparing snapshots: snapshot_name=%s, snapshot_sources=%s, wanted_sources=%s" % (remote_snapshot['Name'], source_snapshots, snapshots_want))
                if snapshots_want == source_snapshots:
                    lg.info("Remote merge snapshot already exists: %s (%s)" % (remote_snapshot['Name'], source_snapshots))
                    self.publish_snapshots.append({
                        'Component': component,
                        'Name': remote_snapshot['Name']
                    })
                    continue

            snapshot_name = '%s%s-%s-%s' % (self.merge_prefix, self.name.replace('./', '').replace('/', '-'), component, self.timestamp)
            lg.info("Creating merge snapshot %s for component %s of snapshots %s" % (snapshot_name, component, snapshots))
            package_refs = []
            for snapshot in snapshots:
                # Get package refs from each snapshot
                packages = self._get_packages(self.client, "snapshots", snapshot)
                package_refs.extend(packages)

            try:
                self.client.do_post(
                    '/snapshots',
                    data={
                        'Name': snapshot_name,
                        'SourceSnapshots': snapshots,
                        'Description': "Merged from sources: %s" % ', '.join("'%s'" % snap for snap in snapshots),
                        'PackageRefs': package_refs,
                    }
                )
            except AptlyException as e:
                if e.res.status_code == 400:
                    lg.warning("Error creating snapshot %s, assuming it already exists" % snapshot_name)
                else:
                    raise

            self.publish_snapshots.append({
                'Component': component,
                'Name': snapshot_name
            })