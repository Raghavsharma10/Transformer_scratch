def query_disease():
    """
    Returns list of diseases by query parameters
    ---

    tags:

      - Query functions

    parameters:

      - name: identifier
        in: query
        type: string
        required: false
        description: Disease identifier
        default: DI-03832

      - name: ref_id
        in: query
        type: string
        required: false
        description: reference identifier
        default: 104300

      - name: ref_type
        in: query
        type: string
        required: false
        description: Reference type
        default: MIM

      - name: name
        in: query
        type: string
        required: false
        description: Disease name
        default: Alzheimer disease

      - name: acronym
        in: query
        type: string
        required: false
        description: Disease acronym
        default: AD

      - name: description
        in: query
        type: string
        required: false
        description: Description of disease
        default: '%neurodegenerative disorder%'

      - name: limit
        in: query
        type: integer
        required: false
        description: limit of results numbers
        default: 10
    """
    allowed_str_args = ['identifier', 'ref_id', 'ref_type', 'name', 'acronym', 'description']

    args = get_args(
        request_args=request.args,
        allowed_str_args=allowed_str_args
    )

    return jsonify(query.disease(**args))