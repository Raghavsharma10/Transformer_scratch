def set_config_value(self, name, value):
        """
        Set configuration value with given name.
        Value can be string or boolean type.
        """
        if value is True:
            value = "True"
        elif value is False:
            if name in self.config_dict:
                del self.config_dict[name]
                self.config_changed = True
            return
        if name not in self.config_dict or self.config_dict[name] != value:
            self.config_changed = True
            self.config_dict[name] = value