def npm(package_json, output_file, pinned_file):
    """Generate a package.json file."""
    amd_build_deprecation_warning()
    try:
        version = get_distribution(current_app.name).version
    except DistributionNotFound:
        version = ''

    output = {
        'name': current_app.name,
        'version': make_semver(version) if version else version,
        'dependencies': {},
    }

    # Load base file
    if package_json:
        output = dict(output, **json.load(package_json))

    # Iterate over bundles
    deps = extract_deps(current_app.extensions['invenio-assets'].env,
                        click.echo)
    output['dependencies'].update(deps)

    # Load pinned dependencies
    if pinned_file:
        output['dependencies'].update(
            json.load(pinned_file).get('dependencies', {}))

    # Write to static folder if output file is not specified
    if output_file is None:
        if not os.path.exists(current_app.static_folder):
            os.makedirs(current_app.static_folder)
        output_file = open(
            os.path.join(current_app.static_folder, 'package.json'),
            'w')

    click.echo('Writing {0}'.format(output_file.name))
    json.dump(output, output_file, indent=4)
    output_file.close()