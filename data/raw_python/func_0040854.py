def do_action(self, command, journal = True):
        """ Implementation for declarative file operations. """

        cmd = 0; src = 1; path = 1; data = 2; dst = 2

        if journal is True:
            self.journal.write(json.dumps(command['undo']) + "\n")
            self.journal.flush()

        d = command['do']
        if   d[cmd] == 'copy':   shutil.copy(d[src], d[dst])
        elif d[cmd] == 'move':   shutil.move(d[src], d[dst])
        elif d[cmd] == 'backup': shutil.move(d[src], self.new_backup(d[src]))
        elif d[cmd] == 'write' :
            if callable(d[data]): d[data](d[path])
            else: file_put_contents(d[path], d[data])