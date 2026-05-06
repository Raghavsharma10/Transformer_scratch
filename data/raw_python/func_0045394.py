def get_summary(session):
    """Get vehicle summary."""
    profile = get_profile(session)
    return {
        'user': {
            'email': profile['userProfile']['eMail'],
            'name': '{} {}'.format(profile['userProfile']['firstName'],
                                   profile['userProfile']['lastName'])
        },
        'vehicles': [
            {
                'vin': vehicle['vin'],
                'year': vehicle['year'],
                'make': vehicle['make'],
                'model': _get_model(vehicle),
                'odometer': vehicle['odometerMileage']
            } for vehicle in profile['vehicles']
        ]
    }