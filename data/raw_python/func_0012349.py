def restore_publish(self, config, components, recreate=False):
        """
        Restore publish from config file
        """
        if "all" in components:
            components = []

        try:
            self.load()
            publish = True
        except NoSuchPublish:
            publish = False

        new_publish_snapshots = []
        to_publish = []
        created_snapshots = []

        for saved_component in config.get('components', []):
            component_name = saved_component.get('component')

            if not component_name:
                raise Exception("Corrupted file")

            if components and component_name not in components:
                continue

            saved_packages = []
            if not saved_component.get('packages'):
                raise Exception("Component %s is empty" % component_name)

            for package in saved_component.get('packages'):
                package_ref = '{} {} {} {}'.format(package.get('arch'), package.get('package'), package.get('version'), package.get('ref'))
                saved_packages.append(package_ref)

            to_publish.append(component_name)

            timestamp = time.strftime("%Y%m%d%H%M%S")
            snapshot_name = '{}-{}-{}'.format("restored", timestamp, saved_component.get('snapshot'))
            lg.debug("Creating snapshot %s for component %s of packages: %s"
                     % (snapshot_name, component_name, saved_packages))

            try:
                self.client.do_post(
                    '/snapshots',
                    data={
                        'Name': snapshot_name,
                        'SourceSnapshots': [],
                        'Description': saved_component.get('description'),
                        'PackageRefs': saved_packages,
                    }
                )
                created_snapshots.append(snapshot_name)
            except AptlyException as e:
                if e.res.status_code == 404:
                    # delete all the previously created
                    # snapshots because the file is corrupted
                    self._remove_snapshots(created_snapshots)
                    raise Exception("Source snapshot or packages don't exist")
                else:
                    raise

            new_publish_snapshots.append({
                'Component': component_name,
                'Name': snapshot_name
            })

        if components:
            self.publish_snapshots = [x for x in self.publish_snapshots if x['Component'] not in components and x['Component'] not in to_publish]
            check_components = [x for x in new_publish_snapshots if x['Component'] in components]
            if len(check_components) != len(components):
                self._remove_snapshots(created_snapshots)
                raise Exception("Not possible to find all the components required in the backup file")

        self.publish_snapshots += new_publish_snapshots
        self.do_publish(recreate=recreate, merge_snapshots=False)