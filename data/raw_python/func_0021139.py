def get_work_kind(self):
        """
        We'll have a kind_slug like 'movies'.
        We need to translate that into a work `kind` like 'movie'.
        """
        slugs_to_kinds = {v:k for k,v in Work.KIND_SLUGS.items()}
        return slugs_to_kinds.get(self.kind_slug, None)