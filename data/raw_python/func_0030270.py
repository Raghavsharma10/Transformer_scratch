def publish_page(page, languages):
    """
    Publish a CMS page in all given languages.
    """
    for language_code, lang_name in iter_languages(languages):
        url = page.get_absolute_url()

        if page.publisher_is_draft:
            page.publish(language_code)
            log.info('page "%s" published in %s: %s', page, lang_name, url)
        else:
            log.info('published page "%s" already exists in %s: %s', page,
                     lang_name, url)
    return page.reload()