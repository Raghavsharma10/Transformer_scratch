def _flush_stack(self):
        '''
        Returns the final output and resets the machine's state.
        '''
        output = self._postprocess_output(''.join(self.stack))
        self._clear_char()
        self._empty_stack()

        if not PYTHON_2:
            return output
        else:
            return unicode(output)