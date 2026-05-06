def get_packages(self, component=None, components=[], packages=None):
        """
        Return package refs for given components
        """
        if component:
            components = [component]

        package_refs = []
        for snapshot in self.publish_snapshots:
            if component and snapshot['Component'] not in components:
                # We don't want packages for this component
                continue

            component_refs = self._get_packages(self.client, "snapshots", snapshot['Name'])
            if packages:
                # Filter package names
                for ref in component_refs:
                    if self.parse_package_ref(ref)[1] in packages:
                        package_refs.append(ref)
            else:
                package_refs.extend(component_refs)

        return package_refs