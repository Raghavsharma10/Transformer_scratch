def render_item(self, contentitem):
        """
        Render the individual item.
        May raise :class:`SkipItem` to ignore an item.
        """
        render_language = get_render_language(contentitem)
        with smart_override(render_language):
            # Plugin output is likely HTML, but it should be placed in mark_safe() to raise awareness about escaping.
            # This is just like Django's Input.render() and unlike Node.render().
            return contentitem.plugin._render_contentitem(self.request, contentitem)