def query_accession():
    """
    Returns list of accession numbers by query query parameters
    ---

    tags:

      - Query functions

    parameters:

      - name: accession
        in: query
        type: string
        required: false
        description: UniProt accession number
        default: P05067

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
        allowed_str_args=['accession', 'entry_name'],
        allowed_int_args=['limit']
    )

    return jsonify(query.accession(**args))