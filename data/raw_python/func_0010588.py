def load(self, name, *, arguments=None, validate_arguments=True, strict_dag=False):
        """ Import the workflow script and load all known objects.

        The workflow script is treated like a module and imported
        into the Python namespace. After the import, the method looks
        for instances of known classes and stores a reference for further
        use in the workflow object.

        Args:
            name (str): The name of the workflow script.
            arguments (dict): Dictionary of additional arguments that are ingested
                              into the data store prior to the execution of the workflow.
            validate_arguments (bool): Whether to check that all required arguments have
                                       been supplied.
            strict_dag (bool): If true then the loaded workflow module must contain an
                               instance of Dag.

        Raises:
            WorkflowArgumentError: If the workflow requires arguments to be set that
                                   were not supplied to the workflow.
            WorkflowImportError: If the import of the workflow fails.
        """
        arguments = {} if arguments is None else arguments

        try:
            workflow_module = importlib.import_module(name)

            dag_present = False

            # extract objects of specific types from the workflow module
            for key, obj in workflow_module.__dict__.items():
                if isinstance(obj, Dag):
                    self._dags_blueprint[obj.name] = obj
                    dag_present = True
                elif isinstance(obj, Parameters):
                    self._parameters.extend(obj)

            self._name = name
            self._docstring = inspect.getdoc(workflow_module)
            del sys.modules[name]

            if strict_dag and not dag_present:
                raise WorkflowImportError(
                    'Workflow does not include a dag {}'.format(name))

            if validate_arguments:
                missing_parameters = self._parameters.check_missing(arguments)
                if len(missing_parameters) > 0:
                    raise WorkflowArgumentError(
                        'The following parameters are required ' +
                        'by the workflow, but are missing: {}'.format(
                            ', '.join(missing_parameters)))

            self._provided_arguments = arguments

        except (TypeError, ImportError):
            logger.error('Cannot import workflow {}'.format(name))
            raise WorkflowImportError('Cannot import workflow {}'.format(name))