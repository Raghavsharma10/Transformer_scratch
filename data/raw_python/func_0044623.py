def process_tags(inst_tags):
    """Create dict of instance tags as only name:value pairs."""
    tag_dict = {}
    for k in range(len(inst_tags)):
        tag_dict[inst_tags[k]['Key']] = inst_tags[k]['Value']
    return tag_dict