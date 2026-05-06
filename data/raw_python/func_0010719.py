def match_tree(sentence, pattern):
    """Matches given sentence with provided pattern.

        :param sentence: sentence from Spacy(see: http://spacy.io/docs/#doc-spans-sents) representing complete statement
        :param pattern: pattern to which sentence will be compared

        :return: True if sentence match to pattern, False otherwise

        :raises: PatternSyntaxException: if pattern has wrong syntax

    """

    if not verify_pattern(pattern):
        raise PatternSyntaxException(pattern)

    def _match_node(t, p):
        pat_node = p.pop(0) if p else ""
        return not pat_node or (_match_token(t, pat_node, False) and _match_edge(t.children,p))

    def _match_edge(edges,p):
        pat_edge = p.pop(0) if p else ""
        if not pat_edge:
            return True
        elif not edges:
            return False
        else:
            for (t) in edges:
                if (_match_token(t, pat_edge, True)) and _match_node(t, list(p)):
                    return True
                elif pat_edge == "**" and _match_edge(t.children, ["**"] + p):
                    return True
        return False
    return _match_node(sentence.root, pattern.split("/"))