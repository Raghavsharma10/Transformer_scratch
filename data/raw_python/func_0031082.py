def query_db_reference():
    """
    Returns list of cross references by query parameters
    ---

    tags:

      - Query functions

    parameters:

      - name: type_
        in: query
        type: string
        required: false
        description: Reference type
        default: EMBL

      - name: identifier
        in: query
        type: string
        required: false
        description: reference identifier
        default: Y00264

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
        allowed_str_args=['type_', 'identifier', 'entry_name'],
        allowed_int_args=['limit']
    )

    return jsonify(query.db_reference(**args))