def get_google_service(service_type=None,version=None):
    '''
    get_url will use the requests library to get a url
    :param service_type: the service to get (default is storage)
    :param version: version to use (default is v1)
    '''
    if service_type == None:
        service_type = "storage"
    if version == None:
        version = "v1"

    credentials = GoogleCredentials.get_application_default()
    return build(service_type, version, credentials=credentials)