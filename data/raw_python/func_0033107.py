def add_key(self):
        "Add ssh key to gitlab if necessary"
        try:
            with open(self.args.ssh_public_key) as f:
                public_key = f.read().strip()
        except:
            log.debug("No key found in {}".format(self.args.ssh_public_key))
            return None
        g = self.gitlab
        url = g['url'] + "/user/keys"
        query = {'private_token': g['token']}
        keys = requests.get(url, params=query).json()
        log.debug("looking for '" + public_key + "' in " + str(keys))
        if (list(filter(lambda key: key['key'] == public_key, keys))):
            log.debug(self.args.ssh_public_key + " already exists")
            return None
        else:
            name = 'github2gitlab'
            log.info("add " + name + " ssh public key from " +
                     self.args.ssh_public_key)
            query['title'] = name
            query['key'] = public_key
            result = requests.post(url, query)
            if result.status_code != requests.codes.created:
                log.warn('Key {} already in GitLab. '
                         'Possible under a different user. Skipping...'
                         .format(self.args.ssh_public_key))
            return public_key