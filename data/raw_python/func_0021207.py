def put(index_name, doc_type, identifier, body, force, verbose):
    """Index input data."""
    result = current_search_client.index(
        index=index_name,
        doc_type=doc_type or index_name,
        id=identifier,
        body=json.load(body),
        op_type='index' if force or identifier is None else 'create',
    )
    if verbose:
        click.echo(json.dumps(result))