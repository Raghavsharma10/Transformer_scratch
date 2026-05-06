def load_entry_point_group_mappings(self, entry_point_group_mappings):
        """Load actions from an entry point group."""
        for ep in iter_entry_points(group=entry_point_group_mappings):
            self.register_mappings(ep.name, ep.module_name)