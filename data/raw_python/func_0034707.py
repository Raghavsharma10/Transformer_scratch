def cleanup_request(request):
        """
        Removes `oauth_` keys from various dicts on the
        request object, and returns the sanitized version.
        """
        for method_type in ('GET', 'PUT', 'POST', 'DELETE'):
            block = getattr(request, method_type, { })

            if True in [ k.startswith("oauth_") for k in block.keys() ]:
                sanitized = block.copy()

                for k in sanitized.keys():
                    if k.startswith("oauth_"):
                        sanitized.pop(k)

                setattr(request, method_type, sanitized)

        return request