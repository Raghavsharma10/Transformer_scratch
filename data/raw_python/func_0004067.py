def get_service_info(self):
        """
        Get information about Workflow Execution Service. May
        include information related (but not limited to) the
        workflow descriptor formats, versions supported, the
        WES API versions supported, and information about general
        the service availability.

        :param str auth: String to send in the auth header.
        :param proto: Schema where the server resides (http, https)
        :param host: Port where the post request will be sent and the wes server listens at (default 8080)
        :return: The body of the get result as a dictionary.
        """
        postresult = requests.get("%s://%s/ga4gh/wes/v1/service-info" % (self.proto, self.host),
                                  headers=self.auth)
        return wes_reponse(postresult)