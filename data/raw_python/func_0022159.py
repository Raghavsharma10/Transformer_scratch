def gravatar(hash, size=100, rating='g', default='identicon', include_extension=False, force_default=False):
        """Pass email hash, return Gravatar URL. You can get email hash like this::

            import hashlib
            avatar_hash = hashlib.md5(email.lower().encode('utf-8')).hexdigest()

        Visit https://en.gravatar.com/site/implement/images/ for more information.

        :param hash: The email hash used to generate avatar URL.
        :param size: The size of the avatar, default to 100 pixel.
        :param rating: The rating of the avatar, default to ``g``
        :param default: The type of default avatar, default to ``identicon``.
        :param include_extension: Append a '.jpg' extension at the end of URL, default to ``False``.
        :param force_default: Force to use default avatar, default to ``False``.
        """
        if include_extension:
            hash += '.jpg'

        default = default or current_app.config['AVATARS_GRAVATAR_DEFAULT']
        query_string = urlencode({'s': int(size), 'r': rating, 'd': default})

        if force_default:
            query_string += '&q=y'
        return 'https://gravatar.com/avatar/' + hash + '?' + query_string