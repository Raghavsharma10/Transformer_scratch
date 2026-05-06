def initialize(template, service_name, environment='dev'):
    """Adds SERVICE_NAME, SERVICE_ENVIRONMENT, and DEFAULT_TAGS to the template

    :param template:
    :param service_name:
    :param environment:
    :return:
    """
    template.SERVICE_NAME = os.getenv('SERVICE_NAME', service_name)
    template.SERVICE_ENVIRONMENT = os.getenv('ENV', environment).lower()
    template.DEFAULT_TAGS = troposphere.Tags(**{
        'service-name': template.SERVICE_NAME,
        'environment': template.SERVICE_ENVIRONMENT
    })
    template.add_version("2010-09-09")
    template.add_description("Stack for %s microservice" % service_name)