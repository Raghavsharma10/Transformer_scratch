def set_config(self, payload):
        """Update the current config depending on the payload and save it."""
        self.config['default'][payload['option']] = str(payload['value'])

        if payload['option'] == 'maxProcesses':
            self.process_handler.set_max(payload['value'])
        if payload['option'] == 'customShell':
            path = payload['value']
            if os.path.isfile(path) and os.access(path, os.X_OK):
                self.process_handler.set_shell(path)
            elif path == 'default':
                self.process_handler.set_shell()
            else:
                return {'message': "File in path doesn't exist or is not executable.",
                        'status': 'error'}

        self.write_config()

        return {'message': 'Configuration successfully updated.',
                'status': 'success'}