def get_run_log(self, run_id):
        """
        Get detailed info about a running workflow.

        :param run_id: String (typically a uuid) identifying the run.
        :param str auth: String to send in the auth header.
        :param proto: Schema where the server resides (http, https)
        :param host: Port where the post request will be sent and the wes server listens at (default 8080)
        :return: The body of the get result as a dictionary.
        """
        postresult = requests.get("%s://%s/ga4gh/wes/v1/runs/%s" % (self.proto, self.host, run_id),
                                  headers=self.auth)
        return wes_reponse(postresult)