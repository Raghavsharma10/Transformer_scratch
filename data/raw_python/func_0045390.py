def get_vehicle_health_report(session, vehicle_index):
    """Get complete vehicle health report."""
    profile = get_profile(session)
    _validate_vehicle(vehicle_index, profile)
    return session.get(VHR_URL, params={
        'uuid': profile['vehicles'][vehicle_index]['uuid']
    }).json()