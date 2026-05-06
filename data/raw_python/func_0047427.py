def _add_arguments(self, args):
        """
        Add command line arguments to each section in config.
        e.x:
        before:
            [global]
            hoge = hoge

        after:
            {
                'global': {
                    'hoge': 'hoge',
                    'arguments': {
                        'debug_mode': True,
                        'and_more': AND_MORE
                    }
                }
            }
        """
        update_dict = {
            'arguments': vars(args)
        }
        for section in self.config.keys():
            self.config[section].update(update_dict)