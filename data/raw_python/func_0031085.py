def query_keyword():
    """
    Returns list of keywords linked to entries by query parameters
    ---
    tags:

      - Query functions

    parameters:

      - name: name
        in: query
        type: string
        required: false
        description: Disease identifier
        default: 'Ubl conjugation'

      - name: identifier
        in: query
        type: string
        required: false
        description: Disease identifier
        default: KW-0832

      - name: entry_name
        in: query
        type: string
        required: false
        description: reference identifier
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
        allowed_str_args=['name', 'identifier', 'entry_name'],
        allowed_int_args=['limit']
    )

    print(args)

    return jsonify(query.keyword(**args))