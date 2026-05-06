def _validate_placement(placement, services, machines, charm, add_error):
    """Validate a placement directive against other services.

    Receive the placement (possibly as a string), the services and machines
    bundle sections, the corresponding charm (or None if invalid) and the
    add_error callable used to register validation errors.

    If applicable, also validate the placement of other machines within the
    bundle.

    Note that some of the logic within this differs between legacy and
    version 4 bundles.

    Return the placement machine id if applicable, None otherwise.
    """
    if not isstring(placement):
        add_error(
            'invalid placement {}: placement must be a string'
            ''.format(placement))
        return
    is_legacy_bundle = machines is None
    try:
        if is_legacy_bundle:
            # This is a v3 legacy bundle.
            unit_placement = models.parse_v3_unit_placement(placement)
            # This is a v4 new style bundle.
        else:
            unit_placement = models.parse_v4_unit_placement(placement)
    except ValueError as e:
        add_error(pyutils.exception_string(e))
        return
    if unit_placement.service:
        service = services.get(unit_placement.service)
        if service is None:
            add_error(
                'placement {} refers to non-existent service {}'
                ''.format(placement, unit_placement.service))
            return
        if unit_placement.unit is not None:
            try:
                num_units = int(service['num_units'])
            except (TypeError, ValueError):
                # This will be notified when validating the service itself.
                pass
            else:
                if int(unit_placement.unit) + 1 > num_units:
                    add_error(
                        'placement {} specifies a unit greater than the units '
                        'in service {}'
                        ''.format(placement, unit_placement.service))
    elif (
        unit_placement.machine and
        not is_legacy_bundle and
        (unit_placement.machine != 'new')
    ):
        machine_id = int(unit_placement.machine)
        # A machine can be included in machines but its value can be None.
        # This is so that we are compatible with go-style YAML unmarshaling.
        if machine_id not in machines:
            add_error(
                'placement {} refers to a non-existent machine {}'
                ''.format(placement, unit_placement.machine))
            return
        machine = machines[machine_id]
        if not isdict(machine):
            # Ignore this error here, as it is emitted while validating the
            # machines section of the bundle.
            machine = {}
        # If the unit is "hulk smashed", then we need to check that the charm
        # and the machine series match.
        if not unit_placement.container_type:
            series = machine.get('series')
            if charm.series and series and charm.series != series:
                # If the machine series is invalid, ignore this check, as an
                # error for the machine will be added elsewhere.
                errors = []
                _validate_series(series, '', errors.append)
                if not errors:
                    add_error(
                        'charm {} cannot be deployed to machine with '
                        'different series {}'.format(charm, series))
        return machine_id