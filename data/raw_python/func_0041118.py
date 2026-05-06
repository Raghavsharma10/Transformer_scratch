def get_location(address=""):
    """
    Retrieve location coordinates from an address introduced.
    """
    coordinates = None
    try:
        geolocator = Nominatim()
        location = geolocator.geocode(address)
        coordinates = (location.latitude, location.longitude)
    except Exception as ex:
        logger.error('Fail get location - {}'.format(ex))
    return coordinates