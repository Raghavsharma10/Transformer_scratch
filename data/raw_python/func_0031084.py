def query_function():
    """
    Returns list of functions by query parameters
    ---
    tags:

      - Query functions

    parameters:

      - name: text
        in: query
        type: string
        required: false
        description: Text describing protein function
        default: '%axonogenesis%'

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
        allowed_str_args=['text', 'entry_name'],
        allowed_int_args=['limit']
    )

    return jsonify(query.function(**args))