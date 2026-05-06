def save(self, new=None, timeout=2):
        """write ALL_VERS_DATA to disk in 'pretty' format"""
        if new: self.update(new) # allow two operations (update + save) with a single command
        if not self._updated: return # nothing to do
        thisPkg = os.path.dirname(__file__)
        filename = os.path.join(thisPkg, c.FOLDER_JSON, c.FILE_GAME_VERSIONS)
        fParts = c.FILE_GAME_VERSIONS.split('.')
        newFile = os.path.join(thisPkg, c.FOLDER_JSON, "%s_%s.%s"%(fParts[0], dateFormat.now(), fParts[1]))
        if not os.path.isfile(newFile):
            #fParts = c.FILE_GAME_VERSIONS.split('.')
            #newFile = "%s%s%s_%s.%s"%(c.FOLDER_JSON, os.sep, fParts[0], dateFormat.now(), fParts[1])
            #if not os.path.isfile(newFile):
            #print(filename)
            #print(newFile)
            os.rename(filename, newFile) # backup existing version file
        recordKeys = [(record["version"], record) for record in Handler.ALL_VERS_DATA.values()]
        data = [r for k,r in sorted(recordKeys)] # i.e. get values sorted by version key
        start = time.time()
        while time.time()-start < timeout: # allow multiple retries if multiple processes fight over the version file
            try:
                with open(filename, "wb") as f:
                    f.write(str.encode(json.dumps(data, indent=4, sort_keys=True))) # python3 requires encoding str => bytes to write to file
                self._updated = False
                return
            except IOError: pass # continue waiting for file to be available
        raise # after timeout, prior exception is what matters