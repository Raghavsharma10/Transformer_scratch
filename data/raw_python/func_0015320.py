def dir_name_changed(self, widget, data=None):
        """
        Function is used for controlling
        label Full Directory project name
        and storing current project directory
        in configuration manager
        """
        config_manager.set_config_value("da.project_dir", self.dir_name.get_text())
        self.update_full_label()