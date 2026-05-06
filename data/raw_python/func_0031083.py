def query_feature():
    """
    Returns list of sequence feature by query parameters
    ---
    tags:

      - Query functions

    parameters:

      - name: type_
        in: query
        type: string
        required: false
        description: Feature type
        default: 'splice variant'

      - name: identifier
        in: query
        type: string
        required: false
        description: Feature identifier
        default: VSP_045447

      - name: description
        in: query
        type: string
        required: false
        description: Feature description
        default: 'In isoform 11.'

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

    return jsonify(query.feature(**args))