def generate_repo_list(product_name=None, product_version=None, product_milestone=None):
    """
    Generates list of artifacts for offline repository.
    """
    if not validate_input_parameters(product_name, product_version, product_milestone):
        sys.exit(1)
    product_version = pnc_api.product_versions.get_all(q="version=='"+ product_version + "';product.name=='"+product_name+"'")
    if not product_version.content:
        logging.error('Specified product version not found.')
        sys.exit(1)
    product_version_id = product_version.content[0].id
    milestone = pnc_api.product_milestones.get_all(q="version=='"+ product_milestone + "';productVersion.id=='"+str(product_version_id)+"'")
    if not milestone.content:
        logging.error('Specified milestone not found.')
        sys.exit(1)
    milestone_id = milestone.content[0].id
    builds = get_all_successful_builds(milestone_id)
    if not builds:
        logging.warning('No builds performed in the milestone.')
        return
    for build in builds:
        built_artifacts = get_all_artifacts(build.id)
        for artifact in built_artifacts:
            print(artifact.identifier)