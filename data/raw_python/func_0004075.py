def run(self, request, tempdir, opts):
        """
        Constructs a command to run a cwl/json from requests and opts,
        runs it, and deposits the outputs in outdir.

        Runner:
        opts.getopt("runner", default="cwl-runner")

        CWL (url):
        request["workflow_url"] == a url to a cwl file
        or
        request["workflow_attachment"] == input cwl text (written to a file and a url constructed for that file)

        JSON File:
        request["workflow_params"] == input json text (to be written to a file)

        :param dict request: A dictionary containing the cwl/json information.
        :param str tempdir: Folder where input files have been staged and the cwd to run at.
        :param wes_service.util.WESBackend opts: contains the user's arguments;
                                                 specifically the runner and runner options
        :return: {"run_id": self.run_id, "state": state}
        """
        wftype = request['workflow_type'].lower().strip()
        version = request['workflow_type_version']

        if version != 'v1.0' and wftype == 'cwl':
            raise RuntimeError('workflow_type "cwl" requires '
                               '"workflow_type_version" to be "v1.0": ' + str(version))
        if version != '2.7' and wftype == 'py':
            raise RuntimeError('workflow_type "py" requires '
                               '"workflow_type_version" to be "2.7": ' + str(version))

        logging.info('Beginning Toil Workflow ID: ' + str(self.run_id))

        with open(self.starttime, 'w') as f:
            f.write(str(time.time()))
        with open(self.request_json, 'w') as f:
            json.dump(request, f)
        with open(self.input_json, "w") as inputtemp:
            json.dump(request["workflow_params"], inputtemp)

        command_args = self.write_workflow(request, opts, tempdir, wftype=wftype)
        pid = self.call_cmd(command_args, tempdir)

        with open(self.endtime, 'w') as f:
            f.write(str(time.time()))
        with open(self.pidfile, 'w') as f:
            f.write(str(pid))

        return self.getstatus()