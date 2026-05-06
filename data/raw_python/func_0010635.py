def list_workflows(config):
    """ List all available workflows.

    Returns a list of all workflows that are available from the paths specified
    in the config. A workflow is defined as a Python file with at least one DAG.

    Args:
        config (Config): Reference to the configuration object from which the
            settings are retrieved.

    Returns:
        list: A list of workflows.
    """
    workflows = []
    for path in config.workflows:
        filenames = glob.glob(os.path.join(os.path.abspath(path), '*.py'))

        for filename in filenames:
            module_name = os.path.splitext(os.path.basename(filename))[0]
            workflow = Workflow()
            try:
                workflow.load(module_name, validate_arguments=False, strict_dag=True)
                workflows.append(workflow)
            except DirectedAcyclicGraphInvalid as e:
                raise WorkflowDefinitionError(workflow_name=module_name,
                                              graph_name=e.graph_name)
            except WorkflowImportError:
                continue

    return workflows