def index(self, request, extra_context=None):
        """
        Displays the dashboard. Includes the main
        navigation that the user has permission for as well
        as the cms log for those sections. The log list can
        be filtered by those same sections
        and is paginated.
        """

        dashboard = self.get_dashboard_urls(request)
        dash_blocks = self.get_dashboard_blocks(request)

        sections, titles = self._get_allowed_sections(dashboard)
        choices = zip(sections, titles)
        choices.sort(key=lambda tup: tup[1])
        choices.insert(0, ('', 'All'))

        class SectionFilterForm(BaseFilterForm):
            section = forms.ChoiceField(required=False, choices=choices)

        form = SectionFilterForm(request.GET)
        filter_kwargs = form.get_filter_kwargs()

        if not filter_kwargs and not request.user.is_superuser:
            filter_kwargs['section__in'] = sections
        cms_logs = models.CMSLog.objects.filter(**filter_kwargs
                                                ).order_by('-when')

        template = self.dashboard_template or 'cms/dashboard.html'

        paginator = Paginator(cms_logs[:20 * 100], 20,
                              allow_empty_first_page=True)
        page_number = request.GET.get('page') or 1
        try:
            page_number = int(page_number)
        except ValueError:
            page_number = 1

        page = paginator.page(page_number)

        return TemplateResponse(request, [template], {
            'dashboard': dashboard, 'blocks': dash_blocks,
            'page': page, 'bundle': self._registry.values()[0],
            'form': form},)