def getTWFiles(self):
        """Get data files with Twitter messages (tweets).
        
        Each item is a list of pickle files on the ../data/tw/ directory
        (probably with some hashtag or common theme).
        Use social.tw.search and social.tw.stream to get tweets.
        Tweets are excluded from package to ease sharing."""

        ddir="../data/tw/"
        # get files in dir
        # order by size, if size
        # group them so that the total filesize is smaller then 1GB-1.4GB
        files=os.path.listdir(ddir)
        files=[i for i in files if os.path.getsize(i)]
        files.sort(key=lambda i: os.path.getsize(i))
        filegroups=self.groupTwitterFilesByEquivalents(files)
        filegroups_grouped=self.groupTwitterFileGroupsForPublishing(filegroups)
        return filegroups_grouped