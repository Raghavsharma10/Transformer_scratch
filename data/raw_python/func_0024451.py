def _rook_legal_for_castle(self, rook):
        """
        Decides if given rook exists, is of this color, and has not moved so it
        is eligible to castle.

        :type: rook: Rook
        :rtype: bool
        """
        return rook is not None and \
            type(rook) is Rook and \
            rook.color == self.color and \
            not rook.has_moved