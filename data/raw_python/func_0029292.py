def analyze(self, scratch, **kwargs):
        """Run and return the results from the DuplicateScripts plugin.

        Only takes into account scripts with more than 3 blocks.

        """
        scripts_set = set()
        for script in self.iter_scripts(scratch):
            if script[0].type.text == 'define %s':
                continue  # Ignore user defined scripts
            blocks_list = []
            for name, _, _ in self.iter_blocks(script.blocks):
                blocks_list.append(name)
            blocks_tuple = tuple(blocks_list)
            if blocks_tuple in scripts_set:
                if len(blocks_list) > 3:
                    self.total_duplicate += 1
                    self.list_duplicate.append(blocks_list)
            else:
                scripts_set.add(blocks_tuple)