def get_address(coords=None, **kwargs):
    """
    Retrieve addres from a location in coords format introduced.
    """
    address = None
    try:
        if (not coords) and \
            ('latitude' in kwargs and 'longitude' in kwargs) or \
                ('location' in kwargs):

            coords = kwargs.get(
                'location', (kwargs.get('latitude'), kwargs.get('longitude')))

        # transform coords
        if isinstance(coords, (list, tuple)) and len(coords) == 2:
            coords = "{}, {}".join(map(str, coords))

        geolocator = Nominatim()
        location = geolocator.reverse(coords)
        address = location.address
    except Exception as ex:
        logger.error('Fail get reverse address - {}'.format(ex))
    return address