def query_ec_number():
    """
    Returns list of Enzyme Commission Numbers (EC numbers) by query parameters
    ---
    tags:

      - Query functions

    parameters:

      - name: ec_number
        in: query
        type: string
        required: false
        description: Enzyme Commission Number
        default: '1.1.1.1'

      - name: entry_name
        in: query
        type: string
        required: false
        description: UniProt entry name
        default: ADHX_HUMAN

      - name: limit
        in: query
        type: integer
        required: false
        description: limit of results numbers
        default: 10
    """
    args = get_args(
        request_args=request.args,
        allowed_str_args=['ec_number', 'entry_name'],
        allowed_int_args=['limit']
    )
    return jsonify(query.ec_number(**args))