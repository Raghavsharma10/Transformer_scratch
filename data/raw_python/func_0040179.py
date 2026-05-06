def install_config_elastic(self):
        """
        install and config elasticsearch
        :return:
        """
        if self.prompt_check("Download and install elasticsearch"):
            self.elastic_install()

        if self.prompt_check("Configure and autostart elasticsearch"):
            self.elastic_config()