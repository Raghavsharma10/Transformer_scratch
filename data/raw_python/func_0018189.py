def get_valid_parent_pages(self, user):
        """
        Identifies possible parent pages for the current user by first looking
        at allowed_parent_page_models() on self.model to limit options to the
        correct type of page, then checking permissions on those individual
        pages to make sure we have permission to add a subpage to it.
        """
        # Start with empty qs
        parents_qs = Page.objects.none()

        # Add pages of the correct type
        for pt in self.model.allowed_parent_page_models():
            pt_items = Page.objects.type(pt)
            parents_qs = parents_qs | pt_items

        # Exclude pages that we can't add subpages to
        for page in parents_qs.all():
            if not page.permissions_for_user(user).can_add_subpage():
                parents_qs = parents_qs.exclude(pk=page.pk)

        return parents_qs