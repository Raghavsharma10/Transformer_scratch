def run(self, wf, jsonyaml, attachments):
        """
        Composes and sends a post request that signals the wes server to run a workflow.

        :param str workflow_file: A local/http/https path to a cwl/wdl/python workflow file.
        :param str jsonyaml: A local path to a json or yaml file.
        :param list attachments: A list of local paths to files that will be uploaded to the server.
        :param str auth: String to send in the auth header.
        :param proto: Schema where the server resides (http, https)
        :param host: Port where the post request will be sent and the wes server listens at (default 8080)

        :return: The body of the post result as a dictionary.
        """
        attachments = list(expand_globs(attachments))
        parts = build_wes_request(wf, jsonyaml, attachments)
        postresult = requests.post("%s://%s/ga4gh/wes/v1/runs" % (self.proto, self.host),
                                   files=parts,
                                   headers=self.auth)
        return wes_reponse(postresult)