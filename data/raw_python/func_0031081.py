def query_organism_host():
    """
    Returns list of host organism by query parameters
    ---

    tags:

      - Query functions

    parameters:

      - name: taxid
        in: query
        type: integer
        required: false
        description: NCBI taxonomy identifier
        default: 9606

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
        allowed_str_args=['entry_name'],
        allowed_int_args=['taxid', 'limit']
    )

    return jsonify(query.organism_host(**args))