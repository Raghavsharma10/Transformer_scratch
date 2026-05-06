def as_yaml(self):
        """
        as_yaml
        """
        self.set_reprdict_from_attributes()
        return "---\n" + yaml.dump(self.m_reprdict, default_flow_style=False)