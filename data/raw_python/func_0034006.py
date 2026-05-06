def get_cache_key(request, meta, orgaMode, currentOrga):
    """Return the cache key to use"""

    # Caching
    cacheKey = None

    if 'cache_time' in meta:
        if meta['cache_time'] > 0:

            # by default, no cache by user
            useUser = False

            # If a logged user in needed, cache the result by user
            if ('only_logged_user' in meta and meta['only_logged_user']) or \
                    ('only_member_user' in meta and meta['only_member_user']) or \
                    ('only_admin_user' in meta and meta['only_admin_user']) or \
                    ('only_orga_member_user' in meta and meta['only_orga_member_user']) or \
                    ('only_orga_admin_user' in meta and meta['only_orga_admin_user']):
                useUser = True

            # If a value if present in meta, use it
            if 'cache_by_user' in meta:
                useUser = meta['cache_by_user']

            cacheKey = '-'

            # Add user info if needed
            if useUser:
                cacheKey += str(request.user.pk) + 'usr-'

            # Add orga
            if orgaMode:
                cacheKey += str(currentOrga.pk) + 'org-'

            # Add current query
            cacheKey += request.get_full_path()

            # Add current template (if the template changed, cache must be invalided)
            cacheKey += meta['template_tag']

    return cacheKey