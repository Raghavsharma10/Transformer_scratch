def get_ref_indices(self):
        """
        :return: list of all indices in object reference.
        """

        ixn_obj = self
        ref_indices = []
        while ixn_obj != ixn_obj.root:
            ref_indices.append(ixn_obj.ref.split(':')[-1])
            ixn_obj = ixn_obj.parent
        return ref_indices[::-1]