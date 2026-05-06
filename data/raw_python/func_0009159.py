def _enclose_readings(container, readings):
    """Enclose a reading within a container, e.g. '[]'."""
    container_start, container_end = tuple(container)
    enclosed_readings = '%(container_start)s%(readings)s%(container_end)s' % {
        'container_start': container_start, 'container_end': container_end,
        'readings': readings}
    return enclosed_readings