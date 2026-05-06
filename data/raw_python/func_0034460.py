def get_traceback_frames(self):
        """Returns the traceback frames as a list"""
        frames = []
        tb = self.tb
        while tb is not None:
            # Support for __traceback_hide__ which is used by a few libraries
            # to hide internal frames.
            if tb.tb_frame.f_locals.get('__traceback_hide__'):
                tb = tb.tb_next
                continue
            filename = tb.tb_frame.f_code.co_filename
            function = tb.tb_frame.f_code.co_name
            lineno = tb.tb_lineno - 1
            loader = tb.tb_frame.f_globals.get('__loader__')
            module_name = tb.tb_frame.f_globals.get('__name__') or ''
            pre_context_lineno, pre_context, context_line, post_context = self._get_lines_from_file(
                filename, lineno, 7, loader, module_name,
            )
            if pre_context_lineno is not None:
                frames.append({
                    'tb': tb,
                    'type': 'django' if module_name.startswith('django.') else 'user',
                    'filename': filename,
                    'function': function,
                    'lineno': lineno + 1,
                    'vars': list(six.iteritems(tb.tb_frame.f_locals)),
                    'id': id(tb),
                    'pre_context': pre_context,
                    'context_line': context_line,
                    'post_context': post_context,
                    'pre_context_lineno': pre_context_lineno + 1,
                })
            tb = tb.tb_next

        return frames