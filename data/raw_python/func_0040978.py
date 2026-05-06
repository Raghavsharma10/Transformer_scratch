def output_files(self):
        """Returns the list of output files from this rule.

        Paths are generated from the outputs of this rule's dependencies, with
        their paths translated based on prefix and strip_prefix.

        Returned paths are relative to buildroot.
        """
        for dep in self.subgraph.successors(self.address):
            dep_rule = self.subgraph.node[dep]['target_obj']
            for dep_file in dep_rule.output_files:
                yield self.translate_path(dep_file, dep_rule).lstrip('/')