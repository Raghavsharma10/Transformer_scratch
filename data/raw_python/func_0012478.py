def _get_bigquery_service(self):
        """
        Connect to the BigQuery service.

        Calling ``GoogleCredentials.get_application_default`` requires that
        you either be running in the Google Cloud, or have the
        ``GOOGLE_APPLICATION_CREDENTIALS`` environment variable set to the path
        to a credentials JSON file.

        :return: authenticated BigQuery service connection object
        :rtype: `googleapiclient.discovery.Resource <http://google.github.io/\
google-api-python-client/docs/epy/googleapiclient.discovery.\
Resource-class.html>`_
        """
        logger.debug('Getting Google Credentials')
        credentials = GoogleCredentials.get_application_default()
        logger.debug('Building BigQuery service instance')
        bigquery_service = build('bigquery', 'v2', credentials=credentials)
        return bigquery_service