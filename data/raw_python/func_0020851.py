def validate(self, data):
        """Check range between dates under keys ``from_`` and ``until``."""
        if 'verb' in data and data['verb'] != self.__class__.__name__:
            raise ValidationError(
                # FIXME encode data
                'This is not a valid OAI-PMH verb:{0}'.format(data['verb']),
                field_names=['verb'],
            )

        if 'from_' in data and 'until' in data and \
                data['from_'] > data['until']:
            raise ValidationError('Date "from" must be before "until".')

        extra = set(request.values.keys()) - set([
            f.load_from or f.name for f in self.fields.values()
        ])
        if extra:
            raise ValidationError('You have passed too many arguments.')