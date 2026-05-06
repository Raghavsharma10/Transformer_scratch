def createRepo(self):
        """
        Creates the repository for all the data we've just downloaded.
        """
        repo = datarepo.SqlDataRepository(self.repoPath)
        repo.open("w")
        repo.initialise()

        referenceSet = references.HtslibReferenceSet("GRCh37-subset")
        referenceSet.populateFromFile(self.fastaFilePath)
        referenceSet.setDescription("Subset of GRCh37 used for demonstration")
        referenceSet.setSpeciesFromJson(
                '{"id": "9606",'
                + '"term": "Homo sapiens", "source_name": "NCBI"}')
        for reference in referenceSet.getReferences():
            reference.setSpeciesFromJson(
                '{"id": "9606",'
                + '"term": "Homo sapiens", "source_name": "NCBI"}')
            reference.setSourceAccessions(
                self.accessions[reference.getName()] + ".subset")
        repo.insertReferenceSet(referenceSet)

        dataset = datasets.Dataset("1kg-p3-subset")
        dataset.setDescription("Sample data from 1000 Genomes phase 3")
        repo.insertDataset(dataset)

        variantSet = variants.HtslibVariantSet(dataset, "mvncall")
        variantSet.setReferenceSet(referenceSet)
        dataUrls = [vcfFile for vcfFile, _ in self.vcfFilePaths]
        indexFiles = [indexFile for _, indexFile in self.vcfFilePaths]
        variantSet.populateFromFile(dataUrls, indexFiles)
        variantSet.checkConsistency()
        repo.insertVariantSet(variantSet)

        for sample, (bamFile, indexFile) in zip(
                self.samples, self.bamFilePaths):
            readGroupSet = reads.HtslibReadGroupSet(dataset, sample)
            readGroupSet.populateFromFile(bamFile, indexFile)
            readGroupSet.setReferenceSet(referenceSet)
            repo.insertReadGroupSet(readGroupSet)

        repo.commit()
        repo.close()
        self.log("Finished creating the repository; summary:\n")
        repo.open("r")
        repo.printSummary()