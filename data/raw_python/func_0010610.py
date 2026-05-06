def workflow_list(obj):
    """ List all available workflows. """
    try:
        for wf in list_workflows(config=obj['config']):
            click.echo('{:23} {}'.format(
                _style(obj['show_color'], wf.name, bold=True),
                wf.docstring.split('\n')[0] if wf.docstring is not None else ''))
    except WorkflowDefinitionError as e:
        click.echo(_style(obj['show_color'],
                          'The graph {} in workflow {} is not a directed acyclic graph'.
                          format(e.graph_name, e.workflow_name), fg='red', bold=True))