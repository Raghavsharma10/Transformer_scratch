def block_type(self):
        """ This gets display on the block header. """
        return capfirst(force_text(
            self.content_block.content_type.model_class()._meta.verbose_name
        ))