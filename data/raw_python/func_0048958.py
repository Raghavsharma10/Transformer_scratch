def _validate_services(services, machines, add_error):
    """Validate each service within the bundle.

    Receive the services and machines sections of the bundle.
    Use the given add_error callable to register validation error.
    """
    machine_ids = set()

    for service_name, service in services.items():
        if not isstring(service_name):
            add_error('service name {} must be a string'.format(service_name))
        if service.get('expose') not in (True, False, None):
            add_error(
                'invalid expose value for service {}'.format(service_name))
        # Validate and retrieve the service charm URL and number of units.
        charm = _validate_charm(service.get('charm'), service_name, add_error)
        num_units = _validate_num_units(
            service.get('num_units'), service_name, add_error)
        # Validate service constraints and storage constraints.
        label = 'service {}'.format(service_name)
        _validate_constraints(service.get('constraints'), label, add_error)
        _validate_storage(service.get('storage'), service_name, add_error)
        # Validate service options and annotations.
        _validate_options(service.get('options'), service_name, add_error)
        _validate_annotations(service.get('annotations'), label, add_error)
        # Retrieve and validate the service units placement.
        placements = service.get('to', [])
        if not islist(placements):
            placements = [placements]
        if (num_units is not None) and (len(placements) > num_units):
            add_error(
                'too many units placed for service {}'.format(service_name))
        for placement in placements:
            machine_id = _validate_placement(
                placement, services, machines, charm, add_error)
            machine_ids.add(machine_id)

    if machines is not None:
        # Notify unused machines.
        unused = set(machines).difference(machine_ids)
        for machine_id in unused:
            add_error(
                'machine {} not referred to by a placement directive'
                ''.format(machine_id))