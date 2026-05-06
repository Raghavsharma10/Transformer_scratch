def retrieve_github_cache(self, github_path, version, cache_dir, token):
        '''
        Retrieves a cache of the layouts git repo from GitHub

        @param github_path: Location of the git repo on GitHub (e.g. hid-io/layouts)
        @param version: git reference for the version to download (e.g. master)
        @param cache_dir: Directory to operate on external cache from
        @param token: GitHub access token
        '''
        # Check for environment variable Github token
        token = os.environ.get('GITHUB_APIKEY', None)

        # Retrieve repo information
        try:
            gh = Github(token)
            repo = gh.get_repo(github_path)
            commit = repo.get_commit(version)
            commits = repo.get_commits()
            total_commits = 0
            commit_number = 0
            for cmt in commits:
                if commit == cmt:
                    commit_number = total_commits
                total_commits += 1
            commit_number = total_commits - commit_number
            tar_url = repo.get_archive_link('tarball', commit.sha)
        except GithubException.RateLimitExceededException:
            if token is None:
                log.warning("GITHUB_APIKEY is not set!")
            raise

        # GitHub only uses the first 7 characters of the sha in the download
        dirname_orig = "{}-{}".format(github_path.replace('/', '-'), commit.sha[:7])
        dirname_orig_path = os.path.join(cache_dir, dirname_orig)
        # Adding a commit number so it's clear which is the latest version without requiring git
        dirname = "{}-{}".format(commit_number, dirname_orig)
        dirname_path = os.path.join(cache_dir, dirname)

        # If directory doesn't exist, check if tarball does
        if not os.path.isdir(dirname_path):
            filename = "{}.tar.gz".format(dirname)
            filepath = os.path.join(cache_dir, filename)

            # If tarball doesn't exist, download it
            if not os.path.isfile(filepath):
                # Retrieve tar file
                chunk_size = 2000
                req = requests.get(tar_url, stream=True)
                with open(filepath, 'wb') as infile:
                    for chunk in req.iter_content(chunk_size):
                        infile.write(chunk)

            # Extract tarfile
            tar = tarfile.open(filepath)
            tar.extractall(cache_dir)
            os.rename(dirname_orig_path, dirname_path)
            tar.close()

            # Remove tar.gz
            os.remove(filepath)