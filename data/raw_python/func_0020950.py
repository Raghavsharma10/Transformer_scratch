def handle_chunk_wrapper(self, status, name, content, file_info):
        """wrapper to allow output redirects for handle_chunk."""
        out = self.output
        if out is not None:
            with out:
                print("handling chunk " + repr(type(content)))
                self.handle_chunk(status, name, content, file_info)
        else:
            self.handle_chunk(status, name, content, file_info)