def kind_name(self):
        "e.g. 'Gig' or 'Movie'."
        return {k:v for (k,v) in self.KIND_CHOICES}[self.kind]