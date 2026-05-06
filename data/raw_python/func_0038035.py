def build_metadata_class(specfile):
    """Generate a metadata class for the specified specfile."""
    with open(specfile) as f:
        spec = json.load(f)
        name = os.path.basename(specfile).split('.')[0]
        spec['name'] = name

        env = get_jinja_env()

        metadata_template = env.get_template('metadata.py.jinja2')
        with open('pycanvas/meta/{}.py'.format(name), 'w') as t:
            t.write(metadata_template.render(spec=spec))