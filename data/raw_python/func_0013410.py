def write_hier_rec(self, gos_printed, out=sys.stdout,
                      len_dash=1, max_depth=None, num_child=None, short_prt=False,
                      include_only=None, go_marks=None,
                      depth=1, dp="-"):
        """Write hierarchy for a GO Term record."""
        # Added by DV Klopfenstein
        GO_id = self.id
        # Shortens hierarchy report by only printing the hierarchy
        # for the sub-set of user-specified GO terms which are connected.
        if include_only is not None and GO_id not in include_only:
          return
        nrp = short_prt and GO_id in gos_printed
        if go_marks is not None:
          out.write('{} '.format('>' if GO_id in go_marks else ' '))
        if len_dash is not None:
            # Default character indicating hierarchy level is '-'.
            # '=' is used to indicate a hierarchical path printed in detail previously.
            letter = '-' if not nrp or not self.children else '='
            dp = ''.join([letter]*depth)
            out.write('{DASHES:{N}} '.format(DASHES=dp, N=len_dash))
        if num_child is not None:
            out.write('{N:>5} '.format(N=len(self.get_all_children())))
        out.write('{GO}\tL-{L:>02}\tD-{D:>02}\t{desc}\n'.format(
            GO=self.id, L=self.level, D=self.depth, desc=self.name))
        # Track GOs previously printed only if needed
        if short_prt:
          gos_printed.add(GO_id)
        # Do not print hierarchy below this turn if it has already been printed
        if nrp:
            return
        depth += 1
        if max_depth is not None and depth > max_depth:
            return
        for p in self.children:
            p.write_hier_rec(gos_printed, out, len_dash, max_depth, num_child, short_prt,
                include_only, go_marks,
                depth, dp)