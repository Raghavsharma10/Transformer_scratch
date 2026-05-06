def check_destination(self, dest, url, rev_options, rev_display):
        """
        Prepare a location to receive a checkout/clone.

        Return True if the location is ready for (and requires) a
        checkout/clone, False otherwise.
        """
        checkout = True
        prompt = False
        if os.path.exists(dest):
            checkout = False
            if os.path.exists(os.path.join(dest, self.dirname)):
                existing_url = self.get_url(dest)
                if self.compare_urls(existing_url, url):
                    logger.info('%s in %s exists, and has correct URL (%s)'
                                % (self.repo_name.title(), display_path(dest), url))
                    logger.notify('Updating %s %s%s'
                                  % (display_path(dest), self.repo_name, rev_display))
                    self.update(dest, rev_options)
                else:
                    logger.warn('%s %s in %s exists with URL %s'
                                % (self.name, self.repo_name, display_path(dest), existing_url))
                    prompt = ('(s)witch, (i)gnore, (w)ipe, (b)ackup ', ('s', 'i', 'w', 'b'))
            else:
                logger.warn('Directory %s already exists, and is not a %s %s.'
                            % (dest, self.name, self.repo_name))
                prompt = ('(i)gnore, (w)ipe, (b)ackup ', ('i', 'w', 'b'))
        if prompt:
            logger.warn('The plan is to install the %s repository %s'
                        % (self.name, url))
            response = ask('What to do?  %s' % prompt[0], prompt[1])

            if response == 's':
                logger.notify('Switching %s %s to %s%s'
                              % (self.repo_name, display_path(dest), url, rev_display))
                self.switch(dest, url, rev_options)
            elif response == 'i':
                # do nothing
                pass
            elif response == 'w':
                logger.warn('Deleting %s' % display_path(dest))
                rmtree(dest)
                checkout = True
            elif response == 'b':
                dest_dir = backup_dir(dest)
                logger.warn('Backing up %s to %s'
                            % (display_path(dest), dest_dir))
                shutil.move(dest, dest_dir)
                checkout = True
        return checkout