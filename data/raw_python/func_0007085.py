def gene_names(self):
        """
        Extract the names of the user-supplied targets
        """
        # Iterate through all the target names in the formatted targets file
        for record in SeqIO.parse(self.targets, 'fasta'):
            # Append all the gene names to the list of names
            self.genes.append(record.id)