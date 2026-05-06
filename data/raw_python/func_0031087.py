def query_subcellular_location():
    """
    Returns list of subcellular locations by query parameters
    ---

    tags:
      - Query functions

    parameters:

      - name: location
        in: query
        type: string
        required: false
        description: Subcellular location
        default: 'Clathrin-coated pit'

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
        allowed_str_args=['location', 'entry_name'],
        allowed_int_args=['limit']
    )

    return jsonify(query.subcellular_location(**args))