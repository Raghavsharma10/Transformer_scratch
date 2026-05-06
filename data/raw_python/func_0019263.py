def remove_linebreaks_within_equations(code):
        r"""Remove line breaks within equations.

        This is not a exhaustive test, but shows how the method works:

        >>> code = 'asdf = \\\n(a\n+b)'
        >>> from hydpy.cythons.modelutils import FuncConverter
        >>> FuncConverter.remove_linebreaks_within_equations(code)
        'asdf = (a+b)'
        """
        code = code.replace('\\\n', '')
        chars = []
        counter = 0
        for char in code:
            if char in ('(', '[', '{'):
                counter += 1
            elif char in (')', ']', '}'):
                counter -= 1
            if not (counter and (char == '\n')):
                chars.append(char)
        return ''.join(chars)