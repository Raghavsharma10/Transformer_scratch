def _validate_machines(machines, add_error):
    """Validate the given machines section.

    Validation includes machines constraints, series and annotations.
    Use the given add_error callable to register validation error.
    """
    if not machines:
        return
    for machine_id, machine in machines.items():
        if machine_id < 0:
            add_error(
                'machine {} has an invalid id, must be positive digit'
                ''.format(machine_id))
        if machine is None:
            continue
        elif not isdict(machine):
            add_error(
                'machine {} does not appear to be well-formed'
                ''.format(machine_id))
            continue
        label = 'machine {}'.format(machine_id)
        _validate_constraints(machine.get('constraints'), label, add_error)
        _validate_series(machine.get('series'), label, add_error)
        _validate_annotations(machine.get('annotations'), label, add_error)