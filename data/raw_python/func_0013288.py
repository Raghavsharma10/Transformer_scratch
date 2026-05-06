def populateFromDirectory(self, vcfDirectory):
        """
        Populates this VariantSet by examing all the VCF files in the
        specified directory. This is mainly used for as a convenience
        for testing purposes.
        """
        pattern = os.path.join(vcfDirectory, "*.vcf.gz")
        dataFiles = []
        indexFiles = []
        for vcfFile in glob.glob(pattern):
            dataFiles.append(vcfFile)
            indexFiles.append(vcfFile + ".tbi")
        self.populateFromFile(dataFiles, indexFiles)