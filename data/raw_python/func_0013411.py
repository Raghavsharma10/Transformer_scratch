def write_hier_all(self, out=sys.stdout,
                      len_dash=1, max_depth=None, num_child=None, short_prt=False):
        """Write hierarchy for all GO Terms in obo file."""
        # Print: [biological_process, molecular_function, and cellular_component]
        for go_id in ['GO:0008150', 'GO:0003674', 'GO:0005575']:
          self.write_hier(go_id, out, len_dash, max_depth, num_child, short_prt, None)