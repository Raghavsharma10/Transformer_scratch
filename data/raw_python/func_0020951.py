def handle_chunk(self, status, name, content, file_info):
        "Handle one chunk of the file.  Override this method for peicewise delivery or error handling."
        if status == "error":
            msg = repr(file_info.get("message"))
            exc = JavaScriptError(msg)
            exc.file_info = file_info
            self.status = "Javascript sent exception " + msg
            self.chunk_collector = []
            raise exc
        if status == "more":
            self.chunk_collector.append(content)
            self.progress_callback(self.chunk_collector, file_info)
        else:
            assert status == "done", "Unknown status " + repr(status)
            self.save_chunks = self.chunk_collector
            self.chunk_collector.append(content)
            all_content = self.combine_chunks(self.chunk_collector)
            self.chunk_collector = []
            content_callback = self.content_callback
            if content_callback is None:
                content_callback = self.default_content_callback
            self.status = "calling " + repr(content_callback)
            try:
                content_callback(self.widget, name, all_content)
            except Exception as e:
                self.status += "\n" + repr(content_callback) + " raised " + repr(e)
                raise