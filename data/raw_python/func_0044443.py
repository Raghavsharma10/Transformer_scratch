def render(template, **kwargs):
    """
    Renders the HTML containing provided summaries.

    The summary has to be an instance of summary.Summary, 
    or at least contain similar properties: title, image, url,
    description and collections: titles, images, descriptions.
    """
    import jinja2
    import os.path as path

    searchpath = path.join(path.dirname(__file__), 
        "templates") 
    loader = jinja2.FileSystemLoader(searchpath=searchpath)
    env = jinja2.Environment(loader=loader)
    temp = env.get_template(template)

    return temp.render(**kwargs)