def _process_extends(self, node, **kwargs):
        """
        Processes an extends block e.g. `{% extends "some/template.jinja" %}`
        """

        # find all the blocks in this template
        for b in self.ast.find_all(nodes.Block):

            # if not already in `child_blocks` then this is the first time a
            # block with this name has been encountered.
            if b.name not in self.child_blocks:
                self.child_blocks[b.name] = b
            else:

                # otherwise we have seen this block before, so we need to find the last
                # super_block and add the block from this template to the end.
                block = self.child_blocks.get(b.name)
                while hasattr(block, 'super_block'):
                    block = block.super_block
                block.super_block = b

        # load the parent template
        parent_template = JinjaToJS(template_root=self.template_root,
                                    template_name=node.template.value,
                                    js_module_format=self.js_module_format,
                                    runtime_path=self.runtime_path,
                                    include_prefix=self.include_prefix,
                                    include_ext=self.include_ext,
                                    child_blocks=self.child_blocks,
                                    dependencies=self.dependencies)

        # add the parent templates output to the current output
        self.output.write(parent_template.output.getvalue())

        # Raise an exception so we stop parsing this template
        raise ExtendsException