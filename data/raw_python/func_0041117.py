def get_country(similar=False, **kwargs):
    """
    Get a country for pycountry
    """
    result_country = None
    try:
        if similar:
            for country in countries:
                if kwargs.get('name', '') in country.name:
                    result_country = country
                    break
        else:
            result_country = countries.get(**kwargs)
    except Exception as ex:
        msg = ('Country not found in pycountry with params introduced'
               ' - {}'.format(ex))
        logger.error(msg, params=kwargs)

    return result_country