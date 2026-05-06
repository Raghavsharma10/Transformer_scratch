def cancel(self, run_id):
        """
        Cancel a running workflow.

        :param run_id: String (typically a uuid) identifying the run.
        :param str auth: String to send in the auth header.
        :param proto: Schema where the server resides (http, https)
        :param host: Port where the post request will be sent and the wes server listens at (default 8080)
        :return: The body of the delete result as a dictionary.
        """
        postresult = requests.post("%s://%s/ga4gh/wes/v1/runs/%s/cancel" % (self.proto, self.host, run_id),
                                   headers=self.auth)
        return wes_reponse(postresult)