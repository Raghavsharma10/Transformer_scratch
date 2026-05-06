def copy_dir(self, source, dest, level=0):
        '''Copies the static files from one directory to another.  If this command is run, we assume the user wants to overwrite any existing files.'''
        encoding = settings.DEFAULT_CHARSET or 'utf8'
        msglevel = 2 if level == 0 else 3
        self.message('Directory: {}'.format(source), msglevel, level)

        # create a directory for this app
        if not os.path.exists(dest):
            self.message('Creating directory: {}'.format(dest), msglevel, level+1)
            os.mkdir(dest)

        # go through the files in this app
        for fname in os.listdir(source):
            source_path = os.path.join(source, fname)
            dest_path = os.path.join(dest, fname)
            ext = os.path.splitext(fname)[1].lower()

            # get the score for this file
            score = 0
            for rule in self.rules:
                score += rule.match(fname, level, TYPE_DIRECTORY if os.path.isdir(source_path) else TYPE_FILE)

            # if score is not above zero, we skip this file
            if score <= 0:
                self.message('Skipping file with score {}: {}'.format(score, source_path), msglevel, level+1)
                continue

            ### if we get here, we need to copy the file ###

            # if a directory, recurse to it
            if os.path.isdir(source_path):
                self.message('Creating directory with score {}: {}'.format(score, source_path), msglevel, level+1)
                # create it in the destination and recurse
                if not os.path.exists(dest_path):
                    os.mkdir(dest_path)
                elif not os.path.isdir(dest_path):  # could be a file or link
                    os.unlink(dest_path)
                    os.mkdir(dest_path)
                self.copy_dir(source_path, dest_path, level+1)

            # if a regular Javscript file, run through the static file processors (scripts group)
            elif ext == '.js' and not self.options.get('no_minify') and jsmin:
                self.message('Including and minifying file with score {}: {}'.format(score, source_path), msglevel, level+1)
                with open(source_path, encoding=encoding) as fin:
                    with open(dest_path, 'w', encoding=encoding) as fout:
                        minified = minify(fin.read(), jsmin)
                        fout.write(minified)


            # if a CSS file, run through the static file processors (styles group)
            elif ext == '.css' and not self.options.get('no_minify') and cssmin:
                self.message('Including and minifying file with score {}: {}'.format(score, source_path), msglevel, level+1)
                with open(source_path, encoding=encoding) as fin:
                    with open(dest_path, 'w', encoding=encoding) as fout:
                        minified = minify(fin.read(), cssmin)
                        fout.write(minified)

            # otherwise, just copy the file
            else:
                self.message('Including file with score {}: {}'.format(score, source_path), msglevel, level+1)
                shutil.copy2(source_path, dest_path)