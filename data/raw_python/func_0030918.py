def progress_callback(self, action, node, elapsed_time=None):
        """
        Callback to report progress

        :param str action:
        :param list node: app, module
        :param int | None elapsed_time:
        """
        if action == 'load_start':
            self.stdout.write('Loading fixture {}.{}...'.format(*node),
                              ending='')
            self.stdout.flush()
        elif action == 'load_success':
            message = 'SUCCESS'
            if elapsed_time:
                message += ' ({:.03} seconds) '.format(elapsed_time)

            self.stdout.write(message)