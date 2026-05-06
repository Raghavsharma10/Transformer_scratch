def save_publish(self, save_path):
        """
        Serialize publish in YAML
        """
        timestamp = time.strftime("%Y%m%d%H%M%S")

        yaml_dict = {}
        yaml_dict["publish"] = self.name
        yaml_dict["name"] = timestamp
        yaml_dict["components"] = []
        yaml_dict["storage"] = self.storage
        for component, snapshots in self.components.items():
            packages = self.get_packages(component)
            package_dict = []
            for package in packages:
                (arch, name, version, ref) = self.parse_package_ref(package)
                package_dict.append({'package': name, 'version': version, 'arch': arch, 'ref': ref})
            snapshot = self._find_snapshot(snapshots[0])

            yaml_dict["components"].append({'component': component, 'snapshot': snapshot['Name'],
                                            'description': snapshot['Description'], 'packages': package_dict})

        name = self.name.replace('/', '-')
        lg.info("Saving publish %s in %s" % (name, save_path))
        with open(save_path, 'w') as save_file:
            yaml.dump(yaml_dict, save_file, default_flow_style=False)