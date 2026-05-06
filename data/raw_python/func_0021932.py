def _vcf_is_strelka(variant_file, variant_metadata):
    """Return True if variant_file given is in strelka format
    """
    if "strelka" in variant_file.lower():
        return True
    elif "NORMAL" in variant_metadata["sample_info"].keys():
        return True
    else:
        vcf_reader = vcf.Reader(open(variant_file, "r"))
        try:
            vcf_type = vcf_reader.metadata["content"]
        except KeyError:
            vcf_type = ""
        if "strelka" in vcf_type.lower():
            return True
    return False