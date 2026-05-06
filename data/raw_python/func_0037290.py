def do_cd(self, path = '/'):
        """change current working directory"""
        path = path[0]

        if path == "..":
            self.current_path = "/".join(self.current_path[:-1].split("/")[0:-1]) + '/'
        elif path == '/':
            self.current_path = "/"
        else:
            if path[-1] == '/':
                self.current_path += path
            else:
                self.current_path += path + '/'

        resp = self.n.getList(self.current_path, type=3)

        if resp:
            self.dirs = []
            self.files = []

            for f in resp:
                name = f['href'].encode('utf-8')

                if name[-1] == '/':
                    self.dirs.append(os.path.basename(name[:-1]))
                else:
                    self.files.append(os.path.basename(name))

        self.prompt = "> %s@Ndrive:%s " %(self.id, self.current_path)