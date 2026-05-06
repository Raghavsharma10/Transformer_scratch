def is_sequential(self):
        """Check if residues that sequence site is composed of are in sequential order.

        :return: If sequence site is in valid sequential order (True) or not (False).
        :rtype: :py:obj:`True` or :py:obj:`False`
        """
        seq_ids = tuple(int(residue["Seq_ID"]) for residue in self)
        return seq_ids == tuple(range(int(seq_ids[0]), int(seq_ids[-1])+1))