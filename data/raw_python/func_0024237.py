def switch_from_external_to_main_wf(self):

        """
        Main workflow switcher.

        This method recreates main workflow from `main wf` dict which
        was set by external workflow swicther previously.

        """

        # in external assigned as True in switch_to_external_wf.
        # external_wf should finish EndEvent and it's name should be
        # also EndEvent for switching again to main wf.
        if self.wf_state['in_external'] and self.current.task_type == 'EndEvent' and \
                self.current.task_name == 'EndEvent':

            # main_wf information was copied in switch_to_external_wf and it takes this information.
            main_wf = self.wf_state['main_wf']

            # main_wf_name is assigned to current workflow name again.
            self.current.workflow_name = main_wf['name']

            # For external WF, check permission and authentication. But after cleaning current task.
            self._clear_current_task()

            # check for auth and perm. current task cleared, do against new workflow_name
            self.check_for_authentication()
            self.check_for_permission()

            # WF knowledge is taken for main wf.
            self.workflow_spec = self.get_worfklow_spec()

            # WF instance is started again where leave off.
            self.workflow = self.deserialize_workflow(main_wf['step'])

            # Current WF is this WF instance.
            self.current.workflow = self.workflow

            # in_external is assigned as False
            self.wf_state['in_external'] = False

            # finished is assigned as False, because still in progress.
            self.wf_state['finished'] = False

            # pool info of main_wf is assigned.
            self.wf_state['pool'] = main_wf['pool']
            self.current.pool = self.wf_state['pool']

            # With main_wf is executed.
            self.run()