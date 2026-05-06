def create_cms_index_pages(placeholder_slot="content"):
    """
    create cms home page and fill >content< placeholder with TextPlugin
    """
    try:
        index_page = Page.objects.get(is_home=True, publisher_is_draft=False)
    except Page.DoesNotExist:
        log.debug('Create index page in "en" and...')

        index_page = create_page(
            title="index in English",
            template=TEMPLATE_INHERITANCE_MAGIC,
            language=settings.LANGUAGE_CODE,
            published=False,
            in_navigation=True)
        placeholder, created = index_page.placeholders.get_or_create(
            slot=placeholder_slot)
        for language_code, lang_name in settings.LANGUAGES:
            with translation.override(language_code):
                title = 'index in %s' % lang_name
                log.info('create %r', title)
                if language_code != settings.LANGUAGE_CODE:
                    create_title(language_code, title, index_page)
                add_plugin(
                    placeholder=placeholder,
                    plugin_type='TextPlugin',  # djangocms_text_ckeditor
                    language=language_code,
                    body='index page in %s' % lang_name)
                index_page.publish(language_code)
        created = True
    else:
        created = False
        log.debug('Index page already exists.')

    return index_page, created