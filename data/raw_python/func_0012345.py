def get_repo_information(config, client, fill_repo=False, components=[]):
        """ fill two dictionnaries : one containing all the packages for every repository
            and the second one associating to every component of every publish its repository"""
        repo_dict = {}
        publish_dict = {}

        for origin in ['repo', 'mirror']:
            for name, repo in config.get(origin, {}).items():
                if components and repo.get('component') not in components:
                    continue
                if fill_repo and origin == 'repo':
                    packages = Publish._get_packages("repos", name)
                    repo_dict[name] = packages
                for distribution in repo.get('distributions'):
                    publish_name = str.join('/', distribution.split('/')[:-1])
                    publish_dict[(publish_name, repo.get('component'))] = name

        return (repo_dict, publish_dict)