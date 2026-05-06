def nodeSatisfiesStringFacet(cntxt: Context, n: Node, nc: ShExJ.NodeConstraint, _c: DebugContext) -> bool:
    """ `5.4.5 XML Schema String Facet Constraints <ttp://shex.io/shex-semantics/#xs-string>`_

     String facet constraints apply to the lexical form of the RDF Literals and IRIs and blank node
     identifiers (see note below regarding access to blank node identifiers).
    """

    # Let lex =
    #
    #  * if the value n is an RDF Literal, the lexical form of the literal (see[rdf11-concepts] section 3.3 Literals).
    #  * if the value n is an IRI, the IRI string (see[rdf11-concepts] section 3.2 IRIs).
    #  * if the value n is a blank node, the blank node identifier (see[rdf11-concepts] section 3.4 Blank Nodes).
    if nc.length is not None or nc.minlength is not None or nc.maxlength is not None \
            or nc.pattern is not None:
        lex = str(n)
        #  Let len = the number of unicode codepoints in lex
        # For a node n and constraint value v, nodeSatisfies(n, v):
        #
        #  * for "length" constraints, v = len,
        #  * for "minlength" constraints, v >= len,
        #  * for "maxlength" constraints, v <= len,
        #  * for "pattern" constraints, v is unescaped into a valid XPath 3.1 regular expression[xpath-functions-31]
        #    re and invoking fn:matches(lex, re) returns fn:true. If the flags parameter is present, it is passed
        #    as a third argument to fn:matches. The pattern may have XPath 3.1 regular expression escape sequences
        #    per the modified production [10] in section 5.6.1.1 as well as numeric escape sequences of the
        #    form 'u' HEX HEX HEX HEX or 'U' HEX HEX HEX HEX HEX HEX HEX HEX. Unescaping replaces numeric escape
        #    sequences with the corresponding unicode codepoint

        # TODO: Figure out whether we need to connect this to the lxml exslt functions
        # TODO: Map flags if not
        if (nc.length is None or len(lex) == nc.length) and \
           (nc.minlength is None or len(lex) >= nc.minlength) and \
           (nc.maxlength is None or len(lex) <= nc.maxlength) and \
           (nc.pattern is None or pattern_match(nc.pattern, nc.flags, lex)):
            return True
        elif nc.length is not None and len(lex) != nc.length:
            cntxt.fail_reason = f"String length mismatch - expected: {nc.length} actual: {len(lex)}"
        elif nc.minlength is not None and len(lex) < nc.minlength:
            cntxt.fail_reason = f"String length violation - minimum: {nc.minlength} actual: {len(lex)}"
        elif nc.maxlength is not None and len(lex) > nc.maxlength:
            cntxt.fail_reason = f"String length violation - maximum: {nc.maxlength} actual: {len(lex)}"
        elif nc.pattern is not None and not pattern_match(nc.pattern, nc.flags, lex):
            cntxt.fail_reason = f"Pattern match failure - pattern: {nc.pattern} flags:{nc.flags}" \
                                             f" string: {lex}"
        else:
            cntxt.fail_reason = "Programming error - flame the programmer"
        return False


    else:
        return True