def generate_plaintext_traceback(self):
        """Like the plaintext attribute but returns a generator"""
        yield text_('Traceback (most recent call last):')
        for frame in self.frames:
            yield text_('  File "%s", line %s, in %s' % (
                frame.filename,
                frame.lineno,
                frame.function_name
                ))
            yield text_('    ' + frame.current_line.strip())
        yield text_(self.exception)