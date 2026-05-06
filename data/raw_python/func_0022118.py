def _wait_for_travis_build(url, commit, committed_at):
    """ Waits for a Travis build to appear with the given commit SHA """
    print('Waiting for a Travis build to appear '
          'for `%s` after `%s`...' % (commit, committed_at))
    import requests

    slug = _slug_from_url(url)
    start_time = time.time()
    build_id = None

    while time.time() - start_time < 60:
        with requests.get('https://api.travis-ci.org/repos/%s/builds' % slug,
                          headers=_travis_headers()) as r:
            if not r.ok:
                raise RuntimeError('Could not reach the Travis API '
                                   'endpoint. Additional information: '
                                   '%s' % str(r.content))

            # Search through all commits and builds to find our build.
            commit_to_sha = {}
            json = r.json()
            for travis_commit in sorted(json['commits'],
                                        key=lambda x: x['committed_at']):
                travis_committed_at = datetime.datetime.strptime(
                    travis_commit['committed_at'], '%Y-%m-%dT%H:%M:%SZ'
                ).replace(tzinfo=utc)
                if travis_committed_at < committed_at:
                    continue
                commit_to_sha[travis_commit['id']] = travis_commit['sha']

            for build in json['builds']:
                if (build['commit_id'] in commit_to_sha and
                        commit_to_sha[build['commit_id']] == commit):

                    build_id = build['id']
                    print('Travis build id: `%d`' % build_id)
                    print('Travis build URL: `https://travis-ci.org/'
                          '%s/builds/%d`' % (slug, build_id))

            if build_id is not None:
                break

        time.sleep(3.0)
    else:
        raise RuntimeError('Timed out while waiting for a Travis build '
                           'to start. Is Travis configured for `%s`?' % url)
    return build_id