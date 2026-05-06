def query_alternative_full_name():
    """
    Returns list of alternative full name by query query parameters
    ---
    tags:

      - Query functions

    parameters:

      - name: name
        in: query
        type: string
        required: false
        description: Alternative full name
        default: 'Alzheimer disease amyloid protein'

      - name: entry_name
        in: query
        type: string
        required: false
        description: UniProt entry name
        default: A4_HUMAN

      - name: limit
        in: query
        type: integer
        required: false
        description: limit of results numbers
        default: 10
    """

    args = get_args(
        request_args=request.args,
        allowed_str_args=['name', 'entry_name'],
        allowed_int_args=['limit']
    )

    return jsonify(query.alternative_full_name(**args))