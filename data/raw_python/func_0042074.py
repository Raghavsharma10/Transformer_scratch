def merge_text_nodes_on(self, node):
        """Merges all consecutive non-translatable text nodes into one"""

        if not isinstance(node, ContainerNode) or not node.children:
            return

        new_children = []
        text_run = []
        for i in node.children:
            if isinstance(i, Text) and not i.translatable:
                text_run.append(i.escaped())
            else:
                if text_run:
                    new_children.append(EscapedText(''.join(text_run)))
                    text_run = []

                new_children.append(i)

        if text_run:
            new_children.append(EscapedText(''.join(text_run)))

        node.children = new_children
        for i in node.children:
            self.merge_text_nodes_on(i)