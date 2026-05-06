def gravatar_url(cls, instance, default="mm", **kwargs):
        """
        returns user gravatar url

        :param instance:
        :param default:
        :param kwargs:
        :return:
        """
        # construct the url
        hash = hashlib.md5(instance.email.encode("utf8").lower()).hexdigest()
        if "d" not in kwargs:
            kwargs["d"] = default
        params = "&".join(
            [
                six.moves.urllib.parse.urlencode({key: value})
                for key, value in kwargs.items()
            ]
        )
        return "https://secure.gravatar.com/avatar/{}?{}".format(hash, params)