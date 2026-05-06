def generate_wf_state_log(self):
        """
        Logs the state of workflow and content of task_data.
        """
        output = '\n- - - - - -\n'
        output += "WORKFLOW: %s ( %s )" % (self.current.workflow_name.upper(),
                                           self.current.workflow.name)

        output += "\nTASK: %s ( %s )\n" % (self.current.task_name, self.current.task_type)
        output += "DATA:"
        for k, v in self.current.task_data.items():
            if v:
                output += "\n\t%s: %s" % (k, v)
        output += "\nCURRENT:"
        output += "\n\tACTIVITY: %s" % self.current.activity
        output += "\n\tPOOL: %s" % self.current.pool
        output += "\n\tIN EXTERNAL: %s" % self.wf_state['in_external']
        output += "\n\tLANE: %s" % self.current.lane_name
        output += "\n\tTOKEN: %s" % self.current.token
        sys._zops_wf_state_log = output
        return output