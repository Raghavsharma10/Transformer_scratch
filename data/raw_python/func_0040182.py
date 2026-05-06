def install_config_spark(self):
        """
        install and config spark
        :return:
        """
        if self.prompt_check("Download and install hadoop"):
            self.hadoop_install()

        if self.prompt_check("Download and install spark"):
            self.spark_install()

        if self.prompt_check("Configure spark"):
            self.spark_config()