def get_html_output(self, result, items):
        """
        Collect all HTML from the rendered items, in the correct ordering.
        The media is also collected in the same ordering, in case it's handled by django-compressor for example.
        """
        html_output = []
        merged_media = Media()
        for contentitem, output in result.get_output(include_exceptions=True):
            if output is ResultTracker.MISSING:
                # Likely get_real_instances() didn't return an item for it.
                # The get_real_instances() didn't return an item for the derived table. This happens when either:
                # 1. that table is truncated/reset, while there is still an entry in the base ContentItem table.
                #    A query at the derived table happens every time the page is being rendered.
                # 2. the model was completely removed which means there is also a stale ContentType object.
                class_name = _get_stale_item_class_name(contentitem)
                html_output.append(mark_safe(u"<!-- Missing derived model for ContentItem #{id}: {cls}. -->\n".format(id=contentitem.pk, cls=class_name)))
                logger.warning("Missing derived model for ContentItem #{id}: {cls}.".format(id=contentitem.pk, cls=class_name))
            elif isinstance(output, Exception):
                html_output.append(u'<!-- error: {0} -->\n'.format(str(output)))
            else:
                html_output.append(output.html)
                add_media(merged_media, output.media)

        return html_output, merged_media