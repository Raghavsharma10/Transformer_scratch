def feed(self):
        "Feed a line from the contents of the GPS log to the daemon."
        line = self.testload.sentences[self.index % len(self.testload.sentences)]
        if "%Delay:" in line:
            # Delay specified number of seconds
            delay = line.split()[1]
            time.sleep(int(delay))
        # self.write has to be set by the derived class
        self.write(line)
        if self.progress:
            self.progress("gpsfake: %s feeds %d=%s\n" % (self.testload.name, len(line), repr(line)))
        time.sleep(WRITE_PAD)
        self.index += 1