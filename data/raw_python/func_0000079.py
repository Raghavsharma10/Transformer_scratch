def get_vector(self, max_choice=3):
        """Return pseudo-choice vectors."""
        vec = {}
        for dim in ['forbidden', 'required', 'permitted']:
            if self.meta[dim] is None:
                continue
            dim_vec = map(lambda x: (x, max_choice), self.meta[dim])
            vec[dim] = dict(dim_vec)
        return vec