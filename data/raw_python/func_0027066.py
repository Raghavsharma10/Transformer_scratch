def changelist_view(self, request, extra_context=None):
        """
        Inject extra links into template context.
        """
        links = []

        for action in self.get_extra_actions():
            links.append({
                'label': self._get_action_label(action),
                'href': self._get_action_href(action)
            })

        extra_context = extra_context or {}
        extra_context['extra_links'] = links

        return super(ExtraActionsMixin, self).changelist_view(
            request, extra_context=extra_context,
        )