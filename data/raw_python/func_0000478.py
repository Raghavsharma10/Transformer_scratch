def default_blocks(self):
        """
        Return a list of default block tuples (appname.ModelName, verbose name).

        Next to the dropdown list of block types, a small number of common blocks which are
        frequently used can be added immediately to a column with one click. This method defines
        the list of default blocks.
        """
        # Use the block list provided by settings if it's defined
        block_list = getattr(settings, 'GLITTER_DEFAULT_BLOCKS', None)

        if block_list is not None:
            return block_list

        # Try and auto fill in default blocks if the apps are installed
        block_list = []

        for block in GLITTER_FALLBACK_BLOCKS:
            app_name, model_name = block.split('.')

            try:
                model_class = apps.get_model(app_name, model_name)
                verbose_name = capfirst(model_class._meta.verbose_name)
                block_list.append((block, verbose_name))
            except LookupError:
                # Block isn't installed - don't add it as a quick add default
                pass

        return block_list