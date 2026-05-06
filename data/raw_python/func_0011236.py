def parse_redirect(redirect, topics, context):
    """Resolve the be.yaml redirect key

    Arguments:
        redirect (dict): Source/destination pairs, e.g. {BE_ACTIVE: ACTIVE}
        topics (tuple): Topics from which to sample, e.g. (project, item, task)
        context (dict): Context from which to sample

    """

    for map_source, map_dest in redirect.items():
        if re.match("{\d+}", map_source):
            topics_index = int(map_source.strip("{}"))
            topics_value = topics[topics_index]
            context[map_dest] = topics_value
            continue

        context[map_dest] = context[map_source]