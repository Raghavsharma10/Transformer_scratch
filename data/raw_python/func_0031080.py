def query_pmid():
    """
    Returns list of PubMed identifier by query parameters
    ---

    tags:

      - Query functions

    parameters:

      - name: pmid
        in: query
        type: string
        required: false
        description: PubMed identifier
        default: 20697050

      - name: entry_name
        in: query
        type: string
        required: false
        description: UniProt entry name
        default: A4_HUMAN

      - name: first
        in: query
        type: string
        required: false
        description: first page
        default: 987

      - name: last
        in: query
        type: string
        required: false
        description: last page
        default: 995

      - name: volume
        in: query
        type: string
        required: false
        description: Volume
        default: 67

      - name: name
        in: query
        type: string
        required: false
        description: Name of journal
        default: 'Arch. Neurol.'

      - name: date
        in: query
        type: string
        required: false
        description: Publication date
        default: 2010

      - name: title
        in: query
        type: string
        required: false
        description: Title of publication
        default: '%amyloidosis%'

      - name: limit
        in: query
        type: integer
        required: false
        description: limit of results numbers
        default: 10
    """
    args = get_args(
        request_args=request.args,
        allowed_str_args=['first', 'last', 'volume', 'name', 'date', 'title', 'entry_name'],
        allowed_int_args=['pmid', 'limit']
    )

    return jsonify(query.pmid(**args))