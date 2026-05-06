def create_logstash(self, **kwargs):
        """
        Creates an instance of the Logging Service.
        """
        logstash = predix.admin.logstash.Logging(**kwargs)
        logstash.create()
        logstash.add_to_manifest(self)

        logging.info('Install Kibana-Me-Logs application by following GitHub instructions')
        logging.info('git clone https://github.com/cloudfoundry-community/kibana-me-logs.git')

        return logstash