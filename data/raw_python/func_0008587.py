def _workaround_no_stage_specific_variables(project):
    """Make Stage-specific variables global (move them to Project)."""
    for (name, var) in project.stage.variables.items():
        yield "variable %s" % name
    for (name, _list) in project.stage.lists.items():
        yield "list %s" % name
    project.variables.update(project.stage.variables)
    project.lists.update(project.stage.lists)
    project.stage.variables = {}
    project.stage.lists = {}