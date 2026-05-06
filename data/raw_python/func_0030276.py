def create_page(self, **extra_kwargs):
        """
        Create page (and page title) in default language

        extra_kwargs will be pass to cms.api.create_page()
        e.g.:
            extra_kwargs={
                "soft_root": True,
                "reverse_id": my_reverse_id,
            }
        """
        with translation.override(self.default_language_code):
            # for evaluate the language name lazy translation
            # e.g.: settings.LANGUAGE_CODE is not "en"

            self.default_lang_name = dict(
                self.languages)[self.default_language_code]
            self.slug = self.get_slug(self.default_language_code,
                                      self.default_lang_name)
            assert self.slug != ""

        page = None
        parent = self.get_parent_page()
        if parent is not None:
            assert parent.publisher_is_draft == True, "Parent page '%s' must be a draft!" % parent

        if self.delete_first:
            if self.apphook_namespace is not None:
                pages = Page.objects.filter(
                    application_namespace=self.apphook_namespace,
                    parent=parent,
                )
            else:
                pages = Page.objects.filter(
                    title_set__slug=self.slug,
                    parent=parent,
                )
            log.debug("Delete %i pages...", pages.count())
            pages.delete()
        else:
            if self.apphook_namespace is not None:
                # Create a plugin page
                queryset = Page.objects.drafts()
                queryset = queryset.filter(parent=parent)
                try:
                    page = queryset.get(
                        application_namespace=self.apphook_namespace)
                except Page.DoesNotExist:
                    pass  # Create page
                else:
                    log.debug("Use existing page: %s", page)
                    created = False
                    return page, created
            else:
                # Not a plugin page
                queryset = Title.objects.filter(
                    language=self.default_language_code)
                queryset = queryset.filter(page__parent=parent)
                try:
                    title = queryset.filter(slug=self.slug).first()
                except Title.DoesNotExist:
                    pass  # Create page
                else:
                    if title is not None:
                        log.debug("Use page from title with slug %r",
                                  self.slug)
                        page = title.page
                        created = False

        if page is None:
            with translation.override(self.default_language_code):
                # set right translation language
                # for evaluate language name lazy translation
                # e.g.: settings.LANGUAGE_CODE is not "en"

                page = create_page(
                    title=self.get_title(self.default_language_code,
                                         self.default_lang_name),
                    menu_title=self.get_menu_title(self.default_language_code,
                                                   self.default_lang_name),
                    template=self.get_template(self.default_language_code,
                                               self.default_lang_name),
                    language=self.default_language_code,
                    slug=self.slug,
                    published=False,
                    parent=parent,
                    in_navigation=self.in_navigation,
                    apphook=self.apphook,
                    apphook_namespace=self.apphook_namespace,
                    **extra_kwargs)
                created = True
                log.debug("Page created in %s: %s", self.default_lang_name,
                          page)

        assert page.publisher_is_draft == True
        return page, created