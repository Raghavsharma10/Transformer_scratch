def process_response(self, request, response):
        """Let's handle old-style response processing here, as usual."""

        # For debug only.
        if not settings.DEBUG:
            return response

        # Check for responses where the data can't be inserted.
        content_encoding = response.get('Content-Encoding', '')
        content_type = response.get('Content-Type', '').split(';')[0]
        if any((getattr(response, 'streaming', False),
                'gzip' in content_encoding,
                content_type not in _HTML_TYPES)):
            return response

        content = force_text(response.content, encoding=settings.DEFAULT_CHARSET)

        pattern = re.escape('</body>')
        bits = re.split(pattern, content, flags=re.IGNORECASE)

        if len(bits) > 1:
            bits[-2] += debug_payload(request, response, self.view_data)
            response.content = "</body>".join(bits)
            if response.get('Content-Length', None):
                response['Content-Length'] = len(response.content)

        return response