def generate_bio_assembly(data_api, struct_inflator):
    """Generate the bioassembly data.
    :param data_api the interface to the decoded data
    :param struct_inflator the interface to put the data into the client object"""
    bioassembly_count = 0
    for bioassembly in data_api.bio_assembly:
        bioassembly_count += 1
        for transform in bioassembly["transformList"]:
            struct_inflator.set_bio_assembly_trans(bioassembly_count,
                                                   transform["chainIndexList"],
                                                   transform["matrix"])