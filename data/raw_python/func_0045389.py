def get_report(session, vehicle_index):
    """Get vehicle health report summary."""
    vhr = get_vehicle_health_report(session, vehicle_index)
    if 'reportCard' not in vhr:
        raise MoparError("no vhr found")
    return _traverse_report(vhr['reportCard'])