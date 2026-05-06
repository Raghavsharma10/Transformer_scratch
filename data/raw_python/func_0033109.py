def unprotect_branches(self):
        "Unprotect branches of the GitLab project"
        g = self.gitlab
        url = g['url'] + "/projects/" + g['repo'] + "/repository/branches"
        query = {'private_token': g['token']}
        unprotected = 0
        r = requests.get(url, params=query)
        r.raise_for_status()
        for branch in r.json():
            if branch['protected']:
                r = requests.put(url + "/" + branch['name'] +
                                 "/unprotect", params=query)
                r.raise_for_status()
                unprotected += 1
        return unprotected