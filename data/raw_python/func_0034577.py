def dd_docs(self):
        """Copy and convert various documentation files."""
        top = os.path.join(os.path.dirname(__file__))
        doc = os.path.join(top, 'doc')

        # Markdown to ronn to man page
        man_md = os.path.join(doc, 'authprogs.md')
        man_ronn = os.path.join(doc, 'authprogs.1.ronn')
        man_1 = os.path.join(doc, 'authprogs.1')

        # Create manpage
        try:
            if not os.path.exists(man_1):
                shutil.copy(man_md, man_ronn)
                self.created.append(man_ronn)
                retval = subprocess.call(['ronn', '-r', man_ronn])
                if retval != 0:
                    raise Exception('ronn man page conversion failed, '
                                    'returned %s' % retval)
                self.created.append(man_1)
        except:
            raise Exception('ronn required for manpage conversion - do you '
                            'have it installed?')

        # Markdown files in docs dir get converted to .html
        for name in MARKDOWN2HTML:
            htmlfile = os.path.join(doc, '%s.html' % name)
            if os.path.exists(htmlfile):
                continue

            target = open(htmlfile, 'w')
            self.created.append(htmlfile)
            stdout = runcmd(['python', '-m', 'markdown',
                             os.path.join(doc, '%s.md' % name)])[1]
            if not stdout:
                raise Exception('markdown conversion failed, no output.')
            target.write(stdout)
            target.close()

        # Markdown files in top level just get renamed sans .md
        for name in MARKDOWN2TEXT:
            target = os.path.join(top, name)
            if os.path.exists(target):
                continue
            source = os.path.join(top, '%s.md' % target)
            shutil.copy(source, target)
            self.created.append(target)