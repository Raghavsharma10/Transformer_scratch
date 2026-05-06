def from_data(cls, type, **data):
        """Create an attachment from data.

        :param str type: attachment type
        :param kwargs data: additional attachment data
        :return: an attachment subclass object
        :rtype: `~groupy.api.attachments.Attachment`
        """
        try:
            return cls._types[type](**data)
        except KeyError:
            return cls(type=type, **data)
        except TypeError as e:
            error = 'could not create {!r} attachment'.format(type)
            raise TypeError('{}: {}'.format(error, e.args[0]))