def add_block_options(self, top):
        """
        Return a list of URLs and titles for blocks which can be added to this column.

        All available blocks are grouped by block category.
        """
        from .blockadmin import blocks

        block_choices = []

        # Group all block by category
        for category in sorted(blocks.site.block_list):
            category_blocks = blocks.site.block_list[category]
            category_choices = []

            for block in category_blocks:
                base_url = reverse('block_admin:{}_{}_add'.format(
                    block._meta.app_label, block._meta.model_name,
                ), kwargs={
                    'version_id': self.glitter_page.version.id,
                })
                block_qs = {
                    'column': self.name,
                    'top': top,
                }
                block_url = '{}?{}'.format(base_url, urlencode(block_qs))
                block_text = capfirst(force_text(block._meta.verbose_name))

                category_choices.append((block_url, block_text))

            category_choices = sorted(category_choices, key=lambda x: x[1])
            block_choices.append((category, category_choices))

        return block_choices