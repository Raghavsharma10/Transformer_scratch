def query_tissue_in_reference():
    """
    Returns list of tissues linked to references by query parameters
    ---

    tags:

      - Query functions

    parameters:
      - name: tissue
        in: query
        type: string
        required: false
        description: Tissue
        default: brain

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
        default: 1
    """
    args = get_args(
        request_args=request.args,
        allowed_str_args=['tissue', 'entry_name'],
        allowed_int_args=['limit']
    )

    return jsonify(query.tissue_in_reference(**args))