def add_project(self):
        "Create project in gitlab if it does not exist"
        g = self.gitlab
        url = g['url'] + "/projects/" + g['repo']
        query = {'private_token': g['token']}
        if (requests.get(url, params=query).status_code == requests.codes.ok):
            log.debug("project " + url + " already exists")
            return None
        else:
            log.info("add project " + g['repo'])
            url = g['url'] + "/projects"
            query['public'] = 'true'
            query['namespace'] = g['namespace']
            query['name'] = g['name']
            result = requests.post(url, params=query)
            if result.status_code != requests.codes.created:
                raise ValueError(result.text)
            log.debug("project " + g['repo'] + " added: " +
                      result.text)
            return result.json()