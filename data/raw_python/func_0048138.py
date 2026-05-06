def parse_v3_unit_placement(placement_str):
    """Return a UnitPlacement for bundles version 3, given a placement string.

    See https://github.com/juju/charmstore/blob/v4/docs/bundles.md
    Raise a ValueError if the placement is not valid.
    """
    placement = placement_str
    container = machine = service = unit = ''
    if ':' in placement:
        try:
            container, placement = placement_str.split(':')
        except ValueError:
            msg = 'placement {} is malformed, too many parts'.format(
                placement_str)
            raise ValueError(msg.encode('utf-8'))
    if '=' in placement:
        try:
            placement, unit = placement.split('=')
        except ValueError:
            msg = 'placement {} is malformed, too many parts'.format(
                placement_str)
            raise ValueError(msg.encode('utf-8'))
    if placement.isdigit():
        machine = placement
    else:
        service = placement
    if (container and container not in VALID_CONTAINERS):
        msg = 'invalid container {} for placement {}'.format(
            container, placement_str)
        raise ValueError(msg.encode('utf-8'))
    unit = _parse_unit(unit, placement_str)
    if machine and machine != '0':
        raise ValueError(b'legacy bundles may not place units on machines '
                         b'other than 0')
    return UnitPlacement(container, machine, service, unit)