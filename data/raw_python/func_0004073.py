def write_workflow(self, request, opts, cwd, wftype='cwl'):
        """Writes a cwl, wdl, or python file as appropriate from the request dictionary."""

        workflow_url = request.get("workflow_url")

        # link the cwl and json into the cwd
        if workflow_url.startswith('file://'):
            os.link(workflow_url[7:], os.path.join(cwd, "wes_workflow." + wftype))
            workflow_url = os.path.join(cwd, "wes_workflow." + wftype)
        os.link(self.input_json, os.path.join(cwd, "wes_input.json"))
        self.input_json = os.path.join(cwd, "wes_input.json")

        extra_options = self.sort_toil_options(opts.getoptlist("extra"))
        if wftype == 'cwl':
            command_args = ['toil-cwl-runner'] + extra_options + [workflow_url, self.input_json]
        elif wftype == 'wdl':
            command_args = ['toil-wdl-runner'] + extra_options + [workflow_url, self.input_json]
        elif wftype == 'py':
            command_args = ['python'] + extra_options + [workflow_url]
        else:
            raise RuntimeError('workflow_type is not "cwl", "wdl", or "py": ' + str(wftype))

        return command_args