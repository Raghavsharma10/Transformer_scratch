def connection_service_name(service, *args):
    ''' the name of a service that manages the connection between services '''
    # if the service is a string
    if isinstance(service, str):
        return service

    return normalize_string(type(service).__name__)