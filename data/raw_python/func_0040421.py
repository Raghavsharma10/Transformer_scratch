def from_string(cls,
                    source,
                    distance_weights=None,
                    merge_same_words=False,
                    group_marker_opening='<<',
                    group_marker_closing='>>'):
        """
        Read a string and derive of ``Graph`` from it.

        Words and punctuation marks are made into nodes.

        Punctuation marks are split into separate nodes unless they fall
        between other non-punctuation marks. ``'hello, world'`` is split
        into ``'hello'``, ``','``, and ``'world'``, while ``'who's there?'``
        is split into ``"who's"``, ``'there'``, and ``'?'``.

        To group arbitrary characters together into a single node
        (e.g. to make ``'hello, world!'``), surround the
        text in question with ``group_marker_opening`` and
        ``group_marker_closing``. With the default value, this
        would look like ``'<<hello, world!>>'``. It is recommended that
        the group markers not appear anywhere in the source text where they
        aren't meant to act as such to prevent unexpected behavior.

        The exact regex for extracting nodes is defined by: ::

            expression = r'{0}(.+){1}|([^\w\s]+)\B|([\S]+\b)'.format(
                ''.join('\\' + c for c in group_marker_opening),
                ''.join('\\' + c for c in group_marker_closing)
            )

        Args:
            source (str): the string to derive the graph from
            distance_weights (dict): dict of relative indices corresponding
                with word weights. For example, if a dict entry is ``1: 1000``
                this means that every word is linked to the word which follows
                it with a weight of 1000. ``-4: 350`` would mean that every
                word is linked to the 4th word behind it with a weight of 350.
                A key of ``0`` refers to the weight words get
                pointing to themselves. Keys pointing beyond the edge of the
                word list will wrap around the list.

                The default value for ``distance_weights`` is ``{1: 1}``.
                This means that each word gets equal weight to whatever
                word follows it. Consequently, if this default value is
                used and ``merge_same_words`` is ``False``, the resulting
                graph behavior will simply move linearly through the
                source, wrapping at the end to the beginning.

            merge_same_words (bool): if nodes which have the same value should
                be merged or not.
            group_marker_opening (str): The string used to mark the beginning
                of word groups.
            group_marker_closing (str): The string used to mark the end
                of word groups. It is strongly recommended that this be
                different than ``group_marker_opening`` to prevent unexpected
                behavior with the regex pattern.

        Returns: Graph

        Example:
            >>> graph = Graph.from_string('i have nothing to say and '
            ...                           'i am saying it and that is poetry.')
            >>> ' '.join(graph.pick().value for i in range(8)) # doctest: +SKIP
            'using chance algorithmic in algorithmic art easier blur'
        """
        if distance_weights is None:
            distance_weights = {1: 1}
        # Convert distance_weights to a sorted list of tuples
        # To make output node list order more predictable
        sorted_weights_list = sorted(distance_weights.items(),
                                     key=lambda i: i[0])
        # regex that matches:
        #   * Anything surrounded by
        #       group_marker_opening and group_marker_closing,
        #   * Groups of punctuation marks followed by whitespace
        #   * Any continuous group of non-whitespace characters
        #       followed by whitespace
        expression = r'{0}(.+){1}|([^\w\s]+)\B|([\S]+\b)'.format(
            ''.join('\\' + c for c in group_marker_opening),
            ''.join('\\' + c for c in group_marker_closing)
        )
        matches = re.findall(expression, source)
        # Un-tuple matches since we are only using groups to strip brackets
        # Is there a better way to do this?
        words = [next(t for t in match if t) for match in matches]

        if merge_same_words:
            # Ensure a 1:1 correspondence between words and nodes,
            # and that all links point to these nodes as well

            # Create nodes for every unique word
            temp_node_list = []
            for word in words:
                if word not in (n.value for n in temp_node_list):
                    temp_node_list.append(Node(word))
            # Loop through words, attaching links to nodes which correspond
            # to the current word. Ensure links also point to valid
            # corresponding nodes in the node list.
            for i, word in enumerate(words):
                matching_node = next(
                    (n for n in temp_node_list if n.value == word))
                for key, weight in sorted_weights_list:
                    # Wrap the index of edge items
                    wrapped_index = (key + i) % len(words)
                    target_word = words[wrapped_index]
                    matching_target_node = next(
                        (n for n in temp_node_list
                         if n.value == target_word))
                    matching_node.add_link(matching_target_node, weight)
        else:
            # Create one node for every (not necessarily unique) word.
            temp_node_list = [Node(word) for word in words]
            for i, node in enumerate(temp_node_list):
                for key, weight in sorted_weights_list:
                    # Wrap the index of edge items
                    wrapped_index = (key + i) % len(temp_node_list)
                    node.add_link(temp_node_list[wrapped_index], weight)

        graph = cls()
        graph.add_nodes(temp_node_list)
        return graph