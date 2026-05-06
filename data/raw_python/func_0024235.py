def start_engine(self, **kwargs):
        """
        Initializes the workflow with given request, response objects and diagram name.

        Args:
            session:
            input:
            workflow_name (str): Name of workflow diagram without ".bpmn" suffix.
             File must be placed under one of configured :py:attr:`~zengine.settings.WORKFLOW_PACKAGES_PATHS`

        """
        self.current = WFCurrent(**kwargs)
        self.wf_state = {'in_external': False, 'finished': False}
        if not self.current.new_token:
            self.wf_state = self.current.wf_cache.get(self.wf_state)
            self.current.workflow_name = self.wf_state['name']
            # if we have a pre-selected object to work with,
            # inserting it as current.input['id'] and task_data['object_id']
            if 'subject' in self.wf_state:
                self.current.input['id'] = self.wf_state['subject']
                self.current.task_data['object_id'] = self.wf_state['subject']
        self.check_for_authentication()
        self.check_for_permission()
        self.workflow = self.load_or_create_workflow()

        # if form data exists in input (user submitted)
        # put form data in wf task_data
        if 'form' in self.current.input:
            form = self.current.input['form']
            if 'form_name' in form:
                self.current.task_data[form['form_name']] = form

        # in wf diagram, if property is stated as init = True
        # demanded initial values are assigned and put to cache
        start_init_values = self.workflow_spec.wf_properties.get('init', 'False') == 'True'
        if start_init_values:
            WFInit = get_object_from_path(settings.WF_INITIAL_VALUES)()
            WFInit.assign_wf_initial_values(self.current)

        log_msg = ("\n\n::::::::::: ENGINE STARTED :::::::::::\n"
                   "\tWF: %s (Possible) TASK:%s\n"
                   "\tCMD:%s\n"
                   "\tSUBCMD:%s" % (
                       self.workflow.name,
                       self.workflow.get_tasks(Task.READY),
                       self.current.input.get('cmd'), self.current.input.get('subcmd')))
        log.debug(log_msg)
        sys._zops_wf_state_log = log_msg
        self.current.workflow = self.workflow