def get_default_blocks(self, top=False):
        """
        Return a list of column default block tuples (URL, verbose name).

        Used for quick add block buttons.
        """
        default_blocks = []

        for block_model, block_name in self.glitter_page.default_blocks:
            block = apps.get_model(block_model)
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

            default_blocks.append((block_url, block_text))

        return default_blocks