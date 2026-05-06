def format_meta_lines(cls, meta, labels, offset, **kwargs):
        '''Return all information from a given meta dictionary in a list of lines'''
        lines = []

        # Name and underline
        name = meta['package_name']
        if 'version' in meta:
            name += '-' + meta['version']
        if 'custom_location' in kwargs:
            name += ' ({loc})'.format(loc=kwargs['custom_location'])

        lines.append(name)
        lines.append(len(name)*'=')
        lines.append('')

        # Summary
        lines.extend(meta['summary'].splitlines())
        lines.append('')

        # Description
        if meta.get('description', ''):
            lines.extend(meta['description'].splitlines())
            lines.append('')


        # Other metadata
        data = []
        for item in labels:
            if meta.get(item, '') != '': # We want to process False and 0
                label = (cls._nice_strings[item] + ':').ljust(offset + 2)
                data.append(label + cls._format_field(meta[item]))

        lines.extend(data)

        return lines