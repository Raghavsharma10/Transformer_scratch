def iter_blocks(block_list):
        """A generator for blocks contained in a block list.

        Yields tuples containing the block name, the depth that the block was
        found at, and finally a handle to the block itself.

        """
        # queue the block and the depth of the block
        queue = [(block, 0) for block in block_list
                 if isinstance(block, kurt.Block)]
        while queue:
            block, depth = queue.pop(0)
            assert block.type.text
            yield block.type.text, depth, block
            for arg in block.args:
                if hasattr(arg, '__iter__'):
                    queue[0:0] = [(x, depth + 1) for x in arg
                                  if isinstance(x, kurt.Block)]
                elif isinstance(arg, kurt.Block):
                    queue.append((arg, depth))