def addVariantSet(self):
        """
        Adds a new VariantSet into this repo.
        """
        self._openRepo()
        dataset = self._repo.getDatasetByName(self._args.datasetName)
        dataUrls = self._args.dataFiles
        name = self._args.name
        if len(dataUrls) == 1:
            if self._args.name is None:
                name = getNameFromPath(dataUrls[0])
            if os.path.isdir(dataUrls[0]):
                # Read in the VCF files from the directory.
                # TODO support uncompressed VCF and BCF files
                vcfDir = dataUrls[0]
                pattern = os.path.join(vcfDir, "*.vcf.gz")
                dataUrls = glob.glob(pattern)
                if len(dataUrls) == 0:
                    raise exceptions.RepoManagerException(
                        "Cannot find any VCF files in the directory "
                        "'{}'.".format(vcfDir))
                dataUrls[0] = self._getFilePath(dataUrls[0],
                                                self._args.relativePath)
        elif self._args.name is None:
            raise exceptions.RepoManagerException(
                "Cannot infer the intended name of the VariantSet when "
                "more than one VCF file is provided. Please provide a "
                "name argument using --name.")
        parsed = urlparse.urlparse(dataUrls[0])
        if parsed.scheme not in ['http', 'ftp']:
            dataUrls = map(lambda url: self._getFilePath(
                url, self._args.relativePath), dataUrls)
        # Now, get the index files for the data files that we've now obtained.
        indexFiles = self._args.indexFiles
        if indexFiles is None:
            # First check if all the paths exist locally, as they must
            # if we are making a default index path.
            for dataUrl in dataUrls:
                if not os.path.exists(dataUrl):
                    raise exceptions.MissingIndexException(
                        "Cannot find file '{}'. All variant files must be "
                        "stored locally if the default index location is "
                        "used. If you are trying to create a VariantSet "
                        "based on remote URLs, please download the index "
                        "files to the local file system and provide them "
                        "with the --indexFiles argument".format(dataUrl))
            # We assume that the indexes are made by adding .tbi
            indexSuffix = ".tbi"
            # TODO support BCF input properly here by adding .csi
            indexFiles = [filename + indexSuffix for filename in dataUrls]
        indexFiles = map(lambda url: self._getFilePath(
            url, self._args.relativePath), indexFiles)
        variantSet = variants.HtslibVariantSet(dataset, name)
        variantSet.populateFromFile(dataUrls, indexFiles)
        # Get the reference set that is associated with the variant set.
        referenceSetName = self._args.referenceSetName
        if referenceSetName is None:
            # Try to find a reference set name from the VCF header.
            referenceSetName = variantSet.getVcfHeaderReferenceSetName()
        if referenceSetName is None:
            raise exceptions.RepoManagerException(
                "Cannot infer the ReferenceSet from the VCF header. Please "
                "specify the ReferenceSet to associate with this "
                "VariantSet using the --referenceSetName option")
        referenceSet = self._repo.getReferenceSetByName(referenceSetName)
        variantSet.setReferenceSet(referenceSet)
        variantSet.setAttributes(json.loads(self._args.attributes))
        # Now check for annotations
        annotationSets = []
        if variantSet.isAnnotated() and self._args.addAnnotationSets:
            ontologyName = self._args.ontologyName
            if ontologyName is None:
                raise exceptions.RepoManagerException(
                    "A sequence ontology name must be provided")
            ontology = self._repo.getOntologyByName(ontologyName)
            self._checkSequenceOntology(ontology)
            for annotationSet in variantSet.getVariantAnnotationSets():
                annotationSet.setOntology(ontology)
                annotationSets.append(annotationSet)

        # Add the annotation sets and the variant set as an atomic update
        def updateRepo():
            self._repo.insertVariantSet(variantSet)
            for annotationSet in annotationSets:
                self._repo.insertVariantAnnotationSet(annotationSet)
        self._updateRepo(updateRepo)