def merge_output(self, result, items, template_name):
        """
        Combine all rendered items. Allow rendering the items with a template,
        to inserting separators or nice start/end code.
        """
        html_output, media = self.get_html_output(result, items)

        if not template_name:
            merged_html = mark_safe(u''.join(html_output))
        else:
            context = {
                'contentitems': list(zip(items, html_output)),
                'parent_object': result.parent_object,  # Can be None
                'edit_mode': self.edit_mode,
            }

            context = PluginContext(self.request, context)
            merged_html = render_to_string(template_name, context.flatten())

        return ContentItemOutput(merged_html, media, cacheable=result.all_cacheable, cache_timeout=result.all_timeout)