def _safe_string(self, source, encoding='utf-8'):
        """Convert unicode to string as gnomekeyring barfs on unicode"""
        if not isinstance(source, str):
            return source.encode(encoding)
        return str(source)