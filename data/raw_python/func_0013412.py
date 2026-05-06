def write_hier(self, GO_id, out=sys.stdout,
                       len_dash=1, max_depth=None, num_child=None, short_prt=False,
                       include_only=None, go_marks=None):
        """Write hierarchy for a GO Term."""
        gos_printed = set()
        self[GO_id].write_hier_rec(gos_printed, out, len_dash, max_depth, num_child,
            short_prt, include_only, go_marks)