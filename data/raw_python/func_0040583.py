def output_files(self):
        """Returns all output files from all of the current module's rules."""
        for dep in self.subgraph.successors(self.address):
            dep_rule = self.subgraph.node[dep]['target_obj']
            for out_file in dep_rule.output_files:
                yield out_file