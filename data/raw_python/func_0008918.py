def _drain_step(self, A, ids, area, done, edge_todo):
        """
        Does a single step of the upstream contributing area calculation.
        Here the pixels in ids are drained downstream, the areas are updated
        and the next set of pixels to drain are determined for the next round.
        """
        # Only drain to cells that have a contribution
        A_todo = A[:, ids.ravel()]
        colsum = np.array(A_todo.sum(1)).ravel()
        # Only touch cells that actually receive a contribution
        # during this stage
        ids_new = colsum != 0
        # Is it possible that I may drain twice from my own cell?
        # -- No, I don't think so...
        # Is it possible that other cells may drain into me in
        # multiple iterations -- yes
        # Then say I check for when I'm done ensures that I don't drain until
        # everyone has drained into me
        area.ravel()[ids_new] += (A_todo[ids_new, :]
                                  * (area.ravel()[ids].ravel()))
        edge_todo.ravel()[ids_new] += (A_todo[ids_new, :]
                                       * (edge_todo.ravel()[ids].ravel()))
        # Figure out what's left to do.
        done.ravel()[ids] = True

        colsum = A * (~done.ravel())
        ids = colsum == 0
        # Figure out the new-undrained ids
        ids = ids & (~done.ravel())

        return ids, area, done, edge_todo