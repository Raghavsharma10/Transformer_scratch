def create_blueprint(state):
    """Create blueprint serving JSON schemas.

    :param state: :class:`invenio_jsonschemas.ext.InvenioJSONSchemasState`
        instance used to retrieve the schemas.
    """
    blueprint = Blueprint(
        'invenio_jsonschemas',
        __name__,
    )

    @blueprint.route('/<path:schema_path>')
    def get_schema(schema_path):
        """Retrieve a schema."""
        try:
            schema_dir = state.get_schema_dir(schema_path)
        except JSONSchemaNotFound:
            abort(404)

        resolved = request.args.get(
            'resolved',
            current_app.config.get('JSONSCHEMAS_RESOLVE_SCHEMA'),
            type=int
        )

        with_refs = request.args.get(
            'refs',
            current_app.config.get('JSONSCHEMAS_REPLACE_REFS'),
            type=int
        ) or resolved

        if resolved or with_refs:
            schema = state.get_schema(
                schema_path,
                with_refs=with_refs,
                resolved=resolved
            )
            return jsonify(schema)
        else:
            return send_from_directory(schema_dir, schema_path)

    return blueprint