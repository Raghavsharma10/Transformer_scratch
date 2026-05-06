def _save_or_delete_workflow(self):
        """
        Calls the real save method if we pass the beggining of the wf
        """
        if not self.current.task_type.startswith('Start'):
            if self.current.task_name.startswith('End') and not self.are_we_in_subprocess():
                self.wf_state['finished'] = True
                self.wf_state['finish_date'] = datetime.now().strftime(
                    settings.DATETIME_DEFAULT_FORMAT)

                if self.current.workflow_name not in settings.EPHEMERAL_WORKFLOWS and not \
                self.wf_state['in_external']:
                    wfi = WFCache(self.current).get_instance()
                    TaskInvitation.objects.filter(instance=wfi, role=self.current.role,
                                              wf_name=wfi.wf.name).delete()

                self.current.log.info("Delete WFCache: %s %s" % (self.current.workflow_name,
                                                                 self.current.token))
            self.save_workflow_to_cache(self.serialize_workflow())