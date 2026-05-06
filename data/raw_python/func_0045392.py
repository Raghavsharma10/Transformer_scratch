def get_tow_guide(session, vehicle_index):
    """Get tow guide information."""
    profile = get_profile(session)
    _validate_vehicle(vehicle_index, profile)
    return session.post(TOW_URL, {
        'vin': profile['vehicles'][vehicle_index]['vin']
    }).json()