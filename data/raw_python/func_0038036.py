def build_model_classes(metadata):
    """Generate a model class for any models contained in the specified spec file."""
    i = importlib.import_module(metadata)
    env = get_jinja_env()
    model_template = env.get_template('model.py.jinja2')
    for model in i.models:
        with open(model_path(model.name.lower()), 'w') as t:
            t.write(model_template.render(model_md=model))