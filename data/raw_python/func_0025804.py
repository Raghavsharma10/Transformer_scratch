def htmlHelp(self, helpString=None, title=None, istask=False, tag=None):
        """ Pop up the help in a browser window.  By default, this tries to
        show the help for the current task.  With the option arguments, it can
        be used to show any help string. """
        # Check the help string.  If it turns out to be a URL, launch that,
        # if not, dump it to a quick and dirty tmp html file to make it
        # presentable, and pass that file name as the URL.
        if not helpString:
            helpString = self.getHelpString(self.pkgName+'.'+self.taskName)
        if not title:
            title = self.taskName
        lwr = helpString.lower()
        if lwr.startswith("http:") or lwr.startswith("https:") or \
           lwr.startswith("file:"):
            url = helpString
            if tag and url.find('#') < 0:
                url += '#'+tag
#           print('LAUNCHING: '+url) # DBG
            irafutils.launchBrowser(url, subj=title)
        else:
            # Write it to a temp HTML file to display
            (fd, fname) = tempfile.mkstemp(suffix='.html', prefix='editpar_')
            os.close(fd)
            f = open(fname, 'w')
            if istask and self._knowTaskHelpIsHtml:
                f.write(helpString)
            else:
                f.write('<html><head><title>'+title+'</title></head>\n')
                f.write('<body><h3>'+title+'</h3>\n')
                f.write('<pre>\n'+helpString+'\n</pre></body></html>')
            f.close()
            irafutils.launchBrowser("file://"+fname, subj=title)