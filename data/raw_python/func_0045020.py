def process_commmon(self):
        '''
        Some data processing common for all services.
        No need to override this.
        '''
        data = self.data
        data_content = data['content'][0]

        ## Paste the output of a command
        # This is deprecated after piping support
        if data['command']:
            try:
                call = subprocess.Popen(data_content.split(),
                                        stderr=subprocess.PIPE,
                                        stdout = subprocess.PIPE)
                out, err = call.communicate()
                content = out
            except OSError:
                logging.exception('Cannot execute the command')
                content = ''

            if not data['title']:
                data['title'] = 'Output of command: `%s`' %(data_content)

        ## Paste the output of a file
        # This is deprecated after piping support
        elif data['file']:
            try:
                f = file(data_content)
                content = f.read()
                f.close()
            except IOError:
                logging.exception('File not present or unreadable')
                content = ''

            if not data['title']:
                data['title'] = 'File: `%s`' %(data_content)
        else:
            content = data_content

        self.data['content'] = content
        self.data['syntax'] = self.SYNTAX_DICT.get(self.data['syntax'], '')

        # Excluded data not useful in paste information
        for key in ['func', 'verbose', 'service', 'extra', 'command', 'file']:
            del self.data[key]