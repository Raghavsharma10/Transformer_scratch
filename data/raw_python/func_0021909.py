def _load_single_patient_merged_variants(self, patient, use_cache=True):
        """ Load merged variants for a single patient, optionally using cache

            Note that merged variants are not filtered.
            Use `_load_single_patient_variants` to get filtered variants
        """
        logger.debug("loading merged variants for patient {}".format(patient.id))
        no_variants = False
        try:
            # get merged-variants from cache
            if use_cache:
                ## load unfiltered variants into list of collections
                variant_cache_file_name = "%s-variants.pkl" % (self.merge_type)
                merged_variants = self.load_from_cache(self.cache_names["variant"], patient.id, variant_cache_file_name)
                if merged_variants is not None:
                    return merged_variants
            # get variant collections from file
            variant_collections = []
            optional_maf_cols = ["t_ref_count", "t_alt_count", "n_ref_count", "n_alt_count"]
            if self.additional_maf_cols is not None:
                optional_maf_cols.extend(self.additional_maf_cols)
            for patient_variants in patient.variants_list:
                if type(patient_variants) == str:
                    if ".vcf" in patient_variants:
                        try:
                            variant_collections.append(varcode.load_vcf_fast(patient_variants))
                        # StopIteration is thrown for empty VCFs. For an empty VCF, don't append any variants,
                        # and don't throw an error. But do record a warning, in case the StopIteration was
                        # thrown for another reason.
                        except StopIteration as e:
                            logger.warning("Empty VCF (or possibly a VCF error) for patient {}: {}".format(
                                patient.id, str(e)))
                    elif ".maf" in patient_variants:
                        # See variant_stats.maf_somatic_variant_stats
                        variant_collections.append(
                            varcode.load_maf(
                                patient_variants,
                                optional_cols=optional_maf_cols,
                                encoding="latin-1"))
                    else:
                        raise ValueError("Don't know how to read %s" % patient_variants)
                elif type(patient_variants) == VariantCollection:
                    variant_collections.append(patient_variants)
                else:
                    raise ValueError("Don't know how to read %s" % patient_variants)
            # merge variant-collections
            if len(variant_collections) == 0:
                no_variants = True
            elif len(variant_collections) == 1:
                # There is nothing to merge
                variants = variant_collections[0]
                merged_variants = variants
            else:
                merged_variants = self._merge_variant_collections(variant_collections, self.merge_type)
        except IOError:
            no_variants = True

        # Note that this is the number of variant collections and not the number of
        # variants. 0 variants will lead to 0 neoantigens, for example, but 0 variant
        # collections will lead to NaN variants and neoantigens.
        if no_variants:
            print("Variants did not exist for patient %s" % patient.id)
            merged_variants = None

        # save merged variants to file
        if use_cache:
            self.save_to_cache(merged_variants, self.cache_names["variant"], patient.id, variant_cache_file_name)
        return merged_variants