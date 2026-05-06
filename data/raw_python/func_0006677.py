def sequence_prep(self):
        """
        Create metadata objects for all PacBio assembly FASTA files in the sequencepath. 
        Create individual subdirectories for each sample. 
        Relative symlink the original FASTA file to the appropriate subdirectory
        """
        # Create a sorted list of all the FASTA files in the sequence path
        strains = sorted(glob(os.path.join(self.fastapath, '*.fa*'.format(self.fastapath))))
        for sample in strains:
            # Create the object
            metadata = MetadataObject()
            # Set the sample name to be the file name of the sequence by removing the path and file extension
            sample_name = os.path.splitext(os.path.basename(sample))[0]
            if sample_name in self.strainset:
                # Extract the OLNID from the dictionary using the SEQID
                samplename = self.straindict[sample_name]
                # samplename = sample_name
                # Set and create the output directory
                outputdir = os.path.join(self.path, samplename)
                make_path(outputdir)
                # Set the name of the JSON file
                json_metadata = os.path.join(outputdir, '{name}.json'.format(name=samplename))
                if not os.path.isfile(json_metadata):
                    # Create the name and output directory attributes
                    metadata.name = samplename
                    metadata.seqid = sample_name
                    metadata.outputdir = outputdir
                    metadata.jsonfile = json_metadata
                    # Set the name of the FASTA file to use in the analyses
                    metadata.bestassemblyfile = os.path.join(metadata.outputdir,
                                                             '{name}.fasta'.format(name=metadata.name))
                    # Symlink the original file to the output directory
                    relative_symlink(sample, outputdir, '{sn}.fasta'.format(sn=metadata.name))
                    # Associate the corresponding FASTQ files with the assembly
                    metadata.fastqfiles = sorted(glob(os.path.join(self.fastqpath,
                                                                   '{name}*.gz'.format(name=metadata.name))))
                    metadata.forward_fastq, metadata.reverse_fastq = metadata.fastqfiles
                    # Write the object to file
                    self.write_json(metadata)
                else:
                    metadata = self.read_json(json_metadata)
                # Add the metadata object to the list of objects
                self.metadata.append(metadata)