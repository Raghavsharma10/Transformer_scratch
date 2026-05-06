def remove_imath_operators(lines):
        """Remove mathematical expressions that require Pythons global
        interpreter locking mechanism.

        This is not a exhaustive test, but shows how the method works:

        >>> lines = ['    x += 1*1']
        >>> from hydpy.cythons.modelutils import FuncConverter
        >>> FuncConverter.remove_imath_operators(lines)
        >>> lines
        ['    x = x + (1*1)']
        """
        for idx, line in enumerate(lines):
            for operator in ('+=', '-=', '**=', '*=', '//=', '/=', '%='):
                sublines = line.split(operator)
                if len(sublines) > 1:
                    indent = line.count(' ') - line.lstrip().count(' ')
                    sublines = [sl.strip() for sl in sublines]
                    line = ('%s%s = %s %s (%s)'
                            % (indent*' ', sublines[0], sublines[0],
                               operator[:-1], sublines[1]))
                    lines[idx] = line