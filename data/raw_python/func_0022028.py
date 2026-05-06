def create_service_from_endpoint(endpoint, service_type, title=None, abstract=None, catalog=None):
    """
    Create a service from an endpoint if it does not already exists.
    """
    from models import Service
    if Service.objects.filter(url=endpoint, catalog=catalog).count() == 0:
        # check if endpoint is valid
        request = requests.get(endpoint)
        if request.status_code == 200:
            LOGGER.debug('Creating a %s service for endpoint=%s catalog=%s' % (service_type, endpoint, catalog))
            service = Service(
                        type=service_type, url=endpoint, title=title, abstract=abstract,
                        csw_type='service', catalog=catalog
                        )
            service.save()
            return service
        else:
            LOGGER.warning('This endpoint is invalid, status code is %s' % request.status_code)
    else:
        LOGGER.warning('A service for this endpoint %s in catalog %s already exists' % (endpoint, catalog))
        return None