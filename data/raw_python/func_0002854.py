def post_gist(report_data, old_sha, new_sha):
    """Post the report to a GitHub Gist and return the URL of the gist."""
    payload = {
        "description": ("Changes in OpenStack-Ansible between "
                        "{0} and {1}".format(old_sha, new_sha)),
        "public": True,
        "files": {
            "osa-diff-{0}-{1}.rst".format(old_sha, new_sha): {
                "content": report_data
            }
        }
    }
    url = "https://api.github.com/gists"
    r = requests.post(url, data=json.dumps(payload))
    response = r.json()
    return response['html_url']