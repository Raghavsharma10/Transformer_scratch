def write(self, writer=None, encoding='utf-8', indent=0, newline='',
            omit_declaration=False, node_depth=0, quote_char='"'):
        """
        Serialize this node and its descendants to text, writing
        the output to a given *writer* or to stdout.

        :param writer: an object such as a file or stream to which XML text
            is sent. If *None* text is sent to :attr:`sys.stdout`.
        :type writer: a file, stream, etc or None
        :param string encoding: the character encoding for serialized text.
        :param indent: indentation prefix to apply to descendent nodes for
            pretty-printing. The value can take many forms:

            - *int*: the number of spaces to indent. 0 means no indent.
            - *string*: a literal prefix for indented nodes, such as ``\\t``.
            - *bool*: no indent if *False*, four spaces indent if *True*.
            - *None*: no indent
        :type indent: string, int, bool, or None
        :param newline: the string value used to separate lines of output.
            The value can take a number of forms:

            - *string*: the literal newline value, such as ``\\n`` or ``\\r``.
              An empty string means no newline.
            - *bool*: no newline if *False*, ``\\n`` newline if *True*.
            - *None*: no newline.
        :type newline: string, bool, or None
        :param boolean omit_declaration: if *True* the XML declaration header
            is omitted, otherwise it is included. Note that the declaration is
            only output when serializing an :class:`xml4h.nodes.Document` node.
        :param int node_depth: the indentation level to start at, such as 2 to
            indent output as if the given *node* has two ancestors.
            This parameter will only be useful if you need to output XML text
            fragments that can be assembled into a document.  This parameter
            has no effect unless indentation is applied.
        :param string quote_char: the character that delimits quoted content.
            You should never need to mess with this.

        Delegates to :func:`xml4h.writer.write_node` applied to this node.
        """
        xml4h.write_node(self,
            writer=writer, encoding=encoding, indent=indent,
            newline=newline, omit_declaration=omit_declaration,
            node_depth=node_depth, quote_char=quote_char)