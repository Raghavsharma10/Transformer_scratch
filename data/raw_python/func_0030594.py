def augment_pipeline(pl, head_pipe=None, tail_pipe=None):
    """
    Augment the pipeline by adding a new pipe section to each stage that has one or more pipes. Can be used for debugging

    :param pl:
    :param DebugPipe:
    :return:
    """

    for k, v in iteritems(pl):
        if v and len(v) > 0:
            if head_pipe and k != 'source':  # Can't put anything before the source.
                v.insert(0, head_pipe)

            if tail_pipe:
                v.append(tail_pipe)