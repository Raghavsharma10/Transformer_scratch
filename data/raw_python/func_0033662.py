def _close(self):
        '''
        Closes aiohttp session and all open file descriptors
        '''
        if hasattr(self, 'aiohttp'):
            if not self.aiohttp.closed:
                self.aiohttp.close()
        if hasattr(self, 'file_descriptors'):
            for fd in self.file_descriptors.values():
                if not fd.closed:
                    fd.close()