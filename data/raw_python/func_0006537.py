def get_gene_seqs(database_path, gene):
    """
    This function takes the database path and a gene name as inputs and 
    returns the gene sequence contained in the file given by the gene name
    """
    gene_path = database_path + "/" + gene + ".fsa"
    gene_seq = ""
    # Open fasta file
    with open(gene_path) as gene_file:
        header = gene_file.readline()
        for line in gene_file:
            seq = line.strip()
            gene_seq += seq
    return gene_seq