def add_tag_for_component(user, c_id):
    """Add a tag on a specific component."""

    v1_utils.verify_existence_and_get(c_id, _TABLE)

    values = {
        'component_id': c_id
    }

    component_tagged = tags.add_tag_to_resource(values,
                                                models.JOIN_COMPONENTS_TAGS)

    return flask.Response(json.dumps(component_tagged), 201,
                          content_type='application/json')