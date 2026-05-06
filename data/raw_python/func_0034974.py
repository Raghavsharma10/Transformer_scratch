def walk(dispatcher, node, definition=None):
    """
    The default, standalone walk function following the standard
    argument ordering for the unparsing walkers.

    Arguments:

    dispatcher
        a Dispatcher instance, defined earlier in this module.  This
        instance will dispatch out the correct callable for the various
        object types encountered throughout this recursive function.

    node
        the starting Node from asttypes.

    definition
        a standalone definition tuple to start working on the node with;
        if none is provided, an initial definition will be looked up
        using the dispatcher with the node for the generation of output.

    While the dispatcher object is able to provide the lookup directly,
    this extra definition argument allow more flexibility in having
    Token subtypes being able to provide specific definitions also that
    may be required, such as the generation of optional rendering
    output.
    """

    # The inner walk function - this is actually exposed to the token
    # rule objects so they can also make use of it to process the node
    # with the dispatcher.

    nodes = []
    sourcepath_stack = [NotImplemented]

    def _walk(dispatcher, node, definition=None, token=None):
        if not isinstance(node, Node):
            for fragment in dispatcher.token(
                    token, nodes[-1], node, sourcepath_stack):
                yield fragment
            return

        push = bool(node.sourcepath)
        if push:
            sourcepath_stack.append(node.sourcepath)
        nodes.append(node)

        if definition is None:
            definition = dispatcher.get_optimized_definition(node)

        for rule in definition:
            for chunk in rule(_walk, dispatcher, node):
                yield chunk

        nodes.pop(-1)
        if push:
            sourcepath_stack.pop(-1)

    # Format layout markers are not handled immediately in the walk -
    # they will simply be buffered so that a collection of them can be
    # handled at once.
    def process_layouts(layout_rule_chunks, last_chunk, chunk):
        before_text = last_chunk.text if last_chunk else None
        after_text = chunk.text if chunk else None
        # the text that was yielded by the previous layout handler
        prev_text = None

        # While Layout rules in a typical definition are typically
        # interspersed with Tokens, certain assumptions with how the
        # Layouts are specified within there will fail when Tokens fail
        # to generate anything for any reason.  However, the dispatcher
        # instance will be able to accept and resolve a tuple of Layouts
        # to some handler function, so that a form of normalization can
        # be done.  For instance, an (Indent, Newline, Dedent) can
        # simply be resolved to no operations.  To achieve this, iterate
        # through the layout_rule_chunks and generate a normalized form
        # for the final handling to happen.

        # the preliminary stack that will be cleared whenever a
        # normalized layout rule chunk is generated.
        lrcs_stack = []

        # first pass: generate both the normalized/finalized lrcs.
        for lrc in layout_rule_chunks:
            lrcs_stack.append(lrc)

            # check every single chunk from left to right...
            for idx in range(len(lrcs_stack)):
                rule = tuple(lrc.rule for lrc in lrcs_stack[idx:])
                handler = dispatcher.layout(rule)
                if handler is not NotImplemented:
                    # not manipulating lrsc_stack from within the same
                    # for loop that it is being iterated upon
                    break
            else:
                # which continues back to the top of the outer for loop
                continue

            # So a handler is found from inside the rules; extend the
            # chunks from the stack that didn't get normalized, and
            # generate a new layout rule chunk.
            lrcs_stack[:] = lrcs_stack[:idx]
            lrcs_stack.append(LayoutChunk(
                rule, handler,
                layout_rule_chunks[idx].node,
            ))

        # second pass: now the processing can be done.
        for lr_chunk in lrcs_stack:
            gen = lr_chunk.handler(
                dispatcher, lr_chunk.node, before_text, after_text, prev_text)
            if not gen:
                continue
            for chunk_from_layout in gen:
                yield chunk_from_layout
                prev_text = chunk_from_layout.text

    # The top level walker implementation
    def walk():
        last_chunk = None
        layout_rule_chunks = []

        for chunk in _walk(dispatcher, node, definition):
            if isinstance(chunk, LayoutChunk):
                layout_rule_chunks.append(chunk)
            else:
                # process layout rule chunks that had been cached.
                for chunk_from_layout in process_layouts(
                        layout_rule_chunks, last_chunk, chunk):
                    yield chunk_from_layout
                layout_rule_chunks[:] = []
                yield chunk
                last_chunk = chunk

        # process the remaining layout rule chunks.
        for chunk_from_layout in process_layouts(
                layout_rule_chunks, last_chunk, None):
            yield chunk_from_layout

    for chunk in walk():
        yield chunk