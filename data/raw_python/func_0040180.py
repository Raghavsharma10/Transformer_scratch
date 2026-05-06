def install_config_logstash(self):
        """
        install and config logstash
        :return:
        """
        if self.prompt_check("Download and install logstash"):
            self.logstash_install()

        if self.prompt_check("Configure and autostart logstash"):
            self.logstash_config()