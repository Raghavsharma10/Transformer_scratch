def AssignVar(self, value):
    """Assign a value to this Value."""
    self.value = value
    # Call OnAssignVar on options.
    [option.OnAssignVar() for option in self.options]