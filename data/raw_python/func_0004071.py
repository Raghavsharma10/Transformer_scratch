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
        :param wes_service.util.WESBackend opts: contains the user's arguments;
                                                 specifically the runner and runner options
        :return: {"run_id": self.run_id, "state": state}
        """
        with open(os.path.join(self.workdir, "request.json"), "w") as f:
            json.dump(request, f)

        with open(os.path.join(self.workdir, "cwl.input.json"), "w") as inputtemp:
            json.dump(request["workflow_params"], inputtemp)

        workflow_url = request.get("workflow_url")  # Will always be local path to descriptor cwl, or url.

        output = open(os.path.join(self.workdir, "cwl.output.json"), "w")
        stderr = open(os.path.join(self.workdir, "stderr"), "w")

        runner = opts.getopt("runner", default="cwl-runner")
        extra = opts.getoptlist("extra")

        # replace any locally specified outdir with the default
        for e in extra:
            if e.startswith('--outdir='):
                extra.remove(e)
        extra.append('--outdir=' + self.outdir)

        # link the cwl and json into the tempdir/cwd
        if workflow_url.startswith('file://'):
            os.symlink(workflow_url[7:], os.path.join(tempdir, "wes_workflow.cwl"))
            workflow_url = os.path.join(tempdir, "wes_workflow.cwl")
        os.symlink(inputtemp.name, os.path.join(tempdir, "cwl.input.json"))
        jsonpath = os.path.join(tempdir, "cwl.input.json")

        # build args and run
        command_args = [runner] + extra + [workflow_url, jsonpath]
        proc = subprocess.Popen(command_args,
                                stdout=output,
                                stderr=stderr,
                                close_fds=True,
                                cwd=tempdir)
        output.close()
        stderr.close()
        with open(os.path.join(self.workdir, "pid"), "w") as pid:
            pid.write(str(proc.pid))

        return self.getstatus()