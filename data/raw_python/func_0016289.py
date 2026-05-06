def get_index_mapping(index):
    """Return the JSON mapping file for an index.

    Mappings are stored as JSON files in the mappings subdirectory of this
    app. They must be saved as {{index}}.json.

    Args:
        index: string, the name of the index to look for.

    """
    # app_path = apps.get_app_config('elasticsearch_django').path
    mappings_dir = get_setting("mappings_dir")
    filename = "%s.json" % index
    path = os.path.join(mappings_dir, filename)
    with open(path, "r") as f:
        return json.load(f)