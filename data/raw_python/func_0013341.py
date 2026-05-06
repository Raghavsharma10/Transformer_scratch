def printSummary(self):
        """
        Prints a summary of this data repository to stdout.
        """
        print("Ontologies:")
        for ontology in self.getOntologys():
            print(
                "",
                ontology.getOntologyPrefix(),
                ontology.getName(),
                ontology.getDataUrl(),
                sep="\t")
        print("ReferenceSets:")
        for referenceSet in self.getReferenceSets():
            print(
                "", referenceSet.getLocalId(), referenceSet.getId(),
                referenceSet.getDescription(), referenceSet.getDataUrl(),
                sep="\t")
            for reference in referenceSet.getReferences():
                print(
                    "\t", reference.getLocalId(), reference.getId(),
                    sep="\t")
        print("Datasets:")
        for dataset in self.getDatasets():
            print(
                "", dataset.getLocalId(), dataset.getId(),
                dataset.getDescription(), sep="\t")
            print("\tReadGroupSets:")
            for readGroupSet in dataset.getReadGroupSets():
                print(
                    "\t", readGroupSet.getLocalId(),
                    readGroupSet.getReferenceSet().getLocalId(),
                    readGroupSet.getId(),
                    readGroupSet.getDataUrl(), sep="\t")
                for readGroup in readGroupSet.getReadGroups():
                    print(
                        "\t\t", readGroup.getId(), readGroup.getLocalId(),
                        sep="\t")
            print("\tVariantSets:")
            for variantSet in dataset.getVariantSets():
                print(
                    "\t", variantSet.getLocalId(),
                    variantSet.getReferenceSet().getLocalId(),
                    variantSet.getId(),
                    sep="\t")
                if variantSet.getNumVariantAnnotationSets() > 0:
                    print("\t\tVariantAnnotationSets:")
                    for vas in variantSet.getVariantAnnotationSets():
                        print(
                            "\t\t", vas.getLocalId(),
                            vas.getAnnotationType(),
                            vas.getOntology().getName(), sep="\t")
            print("\tFeatureSets:")
            for featureSet in dataset.getFeatureSets():
                print(
                    "\t", featureSet.getLocalId(),
                    featureSet.getReferenceSet().getLocalId(),
                    featureSet.getOntology().getName(),
                    featureSet.getId(),
                    sep="\t")
            print("\tContinuousSets:")
            for continuousSet in dataset.getContinuousSets():
                print(
                    "\t", continuousSet.getLocalId(),
                    continuousSet.getReferenceSet().getLocalId(),
                    continuousSet.getId(),
                    sep="\t")
            print("\tPhenotypeAssociationSets:")
            for phenotypeAssociationSet in \
                    dataset.getPhenotypeAssociationSets():
                print(
                    "\t", phenotypeAssociationSet.getLocalId(),
                    phenotypeAssociationSet.getParentContainer().getId(),
                    sep="\t")
                # TODO -  please improve this listing
            print("\tRnaQuantificationSets:")
            for rna_quantification_set in dataset.getRnaQuantificationSets():
                print(
                    "\t", rna_quantification_set.getLocalId(),
                    rna_quantification_set.getId(), sep="\t")
                for quant in rna_quantification_set.getRnaQuantifications():
                        print(
                            "\t\t", quant.getLocalId(),
                            quant._description,
                            ",".join(quant._readGroupIds),
                            ",".join(quant._featureSetIds), sep="\t")