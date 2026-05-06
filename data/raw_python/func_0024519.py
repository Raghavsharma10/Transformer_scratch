def package_in_memory(cls, workflow_name, workflow_files):
        """
        Generates wf packages from workflow diagrams.

        Args:
            workflow_name: Name of wf
            workflow_files:  Diagram  file.

        Returns:
            Workflow package (file like) object
        """
        s = StringIO()
        p = cls(s, workflow_name, meta_data=[])
        p.add_bpmn_files_by_glob(workflow_files)
        p.create_package()
        return s.getvalue()