def format_assistants_lines(cls, assistants):
        '''Return formatted assistants from the given list in human readable form.'''
        lines = cls._format_files(assistants, 'assistants')

        # Assistant help
        if assistants:
            lines.append('')
            assistant = strip_prefix(random.choice(assistants), 'assistants').replace(os.path.sep, ' ').strip()
            if len(assistants) == 1:
                strings = ['After you install this DAP, you can find help about the Assistant',
                           'by running "da {a} -h" .']
            else:
                strings = ['After you install this DAP, you can find help, for example about the Assistant',
                           '"{a}", by running "da {a} -h".']
            lines.extend([l.format(a=assistant) for l in strings])

        return lines