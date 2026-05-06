def import_gene_history(file_handle, tax_id, tax_id_col, id_col, symbol_col):
    """
    Read input gene history file into the database.
    Note that the arguments tax_id_col, id_col and symbol_col have been
    converted into 0-based column indexes.
    """

    # Make sure that tax_id is not "" or "  "
    if not tax_id or tax_id.isspace():
        raise Exception("Input tax_id is blank")

    # Make sure that tax_id exists in Organism table in the database.
    try:
        organism = Organism.objects.get(taxonomy_id=tax_id)
    except Organism.DoesNotExist:
        raise Exception('Input tax_id %s does NOT exist in Organism table. '
                        'Please add it into Organism table first.' % tax_id)

    if tax_id_col < 0 or id_col < 0 or symbol_col < 0:
        raise Exception(
            'tax_id_col, id_col and symbol_col must be positive integers')

    for line_index, line in enumerate(file_handle):
        if line.startswith('#'):  # Skip comment lines.
            continue

        fields = line.rstrip().split('\t')
        # Check input column numbers.
        chk_col_numbers(line_index + 1, len(fields), tax_id_col, id_col,
                        symbol_col)

        # Skip lines whose tax_id's do not match input tax_id.
        if tax_id != fields[tax_id_col]:
            continue

        entrez_id = fields[id_col]
        # If the gene already exists in database, set its "obsolete" attribute
        # to True; otherwise create a new obsolete Gene record in database.
        try:
            gene = Gene.objects.get(entrezid=entrez_id)
            if not gene.obsolete:
                gene.obsolete = True
                gene.save()
        except Gene.DoesNotExist:
            Gene.objects.create(entrezid=entrez_id, organism=organism,
                                systematic_name=fields[symbol_col],
                                obsolete=True)