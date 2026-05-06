def install_config_kafka(self):
        """
        install and config kafka
        :return:
        """
        if self.prompt_check("Download and install kafka"):
            self.kafka_install()

        if self.prompt_check("Configure and autostart kafka"):
            self.kafka_config()