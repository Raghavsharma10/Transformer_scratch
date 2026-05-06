def install_config_kibana(self):
        """
        install and config kibana
        :return:
        """
        if self.prompt_check("Download and install kibana"):
            self.kibana_install()

        if self.prompt_check("Configure and autostart kibana"):
            self.kibana_config()