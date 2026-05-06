def build_service_class(metadata):
    """Generate a service class for the service contained in the specified metadata class."""
    i = importlib.import_module(metadata)
    service = i.service
    env = get_jinja_env()
    service_template = env.get_template('service.py.jinja2')
    with open(api_path(service.name.lower()), 'w') as t:
        t.write(service_template.render(service_md=service))