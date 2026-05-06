def load_entry_point_group_templates(self, entry_point_group_templates):
        """Load actions from an entry point group."""
        result = []
        for ep in iter_entry_points(group=entry_point_group_templates):
            with self.app.app_context():
                for template_dir in ep.load()():
                    result.append(self.register_templates(template_dir))
        return result