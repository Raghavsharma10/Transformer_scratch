def _updateVariantAnnotationSets(self, variantFile, dataUrl):
        """
        Updates the variant annotation set associated with this variant using
        information in the specified pysam variantFile.
        """
        # TODO check the consistency of this between VCF files.
        if not self.isAnnotated():
            annotationType = None
            for record in variantFile.header.records:
                if record.type == "GENERIC":
                    if record.key == "SnpEffVersion":
                        annotationType = ANNOTATIONS_SNPEFF
                    elif record.key == "VEP":
                        version = record.value.split()[0]
                        # TODO we need _much_ more sophisticated processing
                        # of VEP versions here. When do they become
                        # incompatible?
                        if version == "v82":
                            annotationType = ANNOTATIONS_VEP_V82
                        elif version == "v77":
                            annotationType = ANNOTATIONS_VEP_V77
                        else:
                            # TODO raise a proper typed exception there with
                            # the file name as an argument.
                            raise ValueError(
                                "Unsupported VEP version {} in '{}'".format(
                                    version, dataUrl))
            if annotationType is None:
                infoKeys = variantFile.header.info.keys()
                if 'CSQ' in infoKeys or 'ANN' in infoKeys:
                    # TODO likewise, we want a properly typed exception that
                    # we can throw back to the repo manager UI and display
                    # as an import error.
                    raise ValueError(
                        "Unsupported annotations in '{}'".format(dataUrl))
            if annotationType is not None:
                vas = HtslibVariantAnnotationSet(self, self.getLocalId())
                vas.populateFromFile(variantFile, annotationType)
                self.addVariantAnnotationSet(vas)