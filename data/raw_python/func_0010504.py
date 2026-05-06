def _process_block(self, node, **kwargs):
        """
        Processes a block e.g. `{% block my_block %}{% endblock %}`
        """

        # check if this node already has a 'super_block' attribute
        if not hasattr(node, 'super_block'):

            # since it doesn't it must be the last block in the inheritance chain
            node.super_block = None

            # see if there has been a child block defined - if there is this
            # will be the first block in the inheritance chain
            child_block = self.child_blocks.get(node.name)

            if child_block:

                # we have child nodes so we need to set `node` as the
                # super of the last one in the chain
                last_block = child_block
                while hasattr(last_block, 'super_block'):
                    last_block = child_block.super_block

                # once we have found it, set this node as it's super block
                last_block.super_block = node

                # this is the node we want to process as it's the first in the inheritance chain
                node = child_block

        # process the block passing the it's super along, if this block
        # calls super() it will be handled by `_process_call`
        for n in node.body:
            self._process_node(n, super_block=node.super_block, **kwargs)