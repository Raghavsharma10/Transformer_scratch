def refresh_role(self, role, file_hierarchy):
        """Checks and refreshes (if needed) all assistants with given role.

        Args:
            role: role of assistants to refresh
            file_hierarchy: hierarchy as returned by devassistant.yaml_assistant_loader.\
                            YamlAssistantLoader.get_assistants_file_hierarchy
        """
        if role not in self.cache:
            self.cache[role] = {}
        was_change = self._refresh_hierarchy_recursive(self.cache[role], file_hierarchy)
        if was_change:
            cf = open(self.cache_file, 'w')
            yaml.dump(self.cache, cf, Dumper=Dumper)
            cf.close()