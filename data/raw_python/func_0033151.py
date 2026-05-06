def _get_result_paths(self, data):
        """ Set the result paths """

        result = {}

        # get the file extension of the reads file (sortmerna
        # internally outputs all results with this extension)
        fileExtension = splitext(self.Parameters['--reads'].Value)[1]

        # at this point the parameter --aligned should be set as
        # sortmerna will not run without it
        if self.Parameters['--aligned'].isOff():
            raise ValueError("Error: the --aligned parameter must be set.")

        # file base name for aligned reads
        output_base = self.Parameters['--aligned'].Value

        # Blast alignments
        result['BlastAlignments'] =\
            ResultPath(Path=output_base + '.blast',
                       IsWritten=self.Parameters['--blast'].isOn())

        # SAM alignments
        result['SAMAlignments'] =\
            ResultPath(Path=output_base + '.sam',
                       IsWritten=self.Parameters['--sam'].isOn())

        # OTU map (mandatory output)
        result['OtuMap'] =\
            ResultPath(Path=output_base + '_otus.txt',
                       IsWritten=self.Parameters['--otu_map'].isOn())

        # FASTA file of sequences in the OTU map (madatory output)
        result['FastaMatches'] =\
            ResultPath(Path=output_base + fileExtension,
                       IsWritten=self.Parameters['--fastx'].isOn())

        # FASTA file of sequences not in the OTU map (mandatory output)
        result['FastaForDenovo'] =\
            ResultPath(Path=output_base + '_denovo' +
                       fileExtension,
                       IsWritten=self.Parameters['--de_novo_otu'].isOn())
        # Log file
        result['LogFile'] =\
            ResultPath(Path=output_base + '.log',
                       IsWritten=self.Parameters['--log'].isOn())

        return result