def query_disease_comment():
    """
    Returns list of diseases comments by query parameters
    ---

    tags:

      - Query functions

    parameters:

      - name: comment
        in: query
        type: string
        required: false
        description: Comment on disease linked to UniProt entry
        default: '%mutations%'

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
        allowed_str_args=['comment', 'entry_name'],
        allowed_int_args=['limit']
    )
    return jsonify(query.disease_comment(**args))