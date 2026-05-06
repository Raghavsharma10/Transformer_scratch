def fasta_records(files):
    """
    Use SeqIO to create dictionaries of all records for each FASTA file
    :param files: dictionary of stain name: /sequencepath/strain_name.extension
    :return: file_records: dictionary of all contig records for all strains
    """
    # Initialise the dictionary
    file_records = dict()
    for file_name, fasta in files.items():
        # Create a dictionary of records for each file
        record_dict = SeqIO.to_dict(SeqIO.parse(fasta, "fasta"))
        # Set the records dictionary as the value for file_records
        file_records[file_name] = record_dict
    return file_records