def get_worfklow_spec(self):
        """
        Generates and caches the workflow spec package from
        BPMN diagrams that read from disk

        Returns:
            SpiffWorkflow Spec object.
        """
        # TODO: convert from in-process to redis based caching
        if self.current.workflow_name not in self.workflow_spec_cache:
            # path = self.find_workflow_path()
            # spec_package = InMemoryPackager.package_in_memory(self.current.workflow_name, path)
            # spec = BpmnSerializer().deserialize_workflow_spec(spec_package)

            try:
                self.current.wf_object = BPMNWorkflow.objects.get(name=self.current.workflow_name)
            except ObjectDoesNotExist:
                self.current.wf_object = BPMNWorkflow.objects.get(name='not_found')
                self.current.task_data['non-existent-wf'] = self.current.workflow_name
                self.current.workflow_name = 'not_found'
            xml_content = self.current.wf_object.xml.body
            spec = ZopsSerializer().deserialize_workflow_spec(xml_content, self.current.workflow_name)

            spec.wf_id = self.current.wf_object.key
            self.workflow_spec_cache[self.current.workflow_name] = spec
        return self.workflow_spec_cache[self.current.workflow_name]