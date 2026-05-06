def get_content(self):
        """Get content of the page through url."""
        url = self.build_url()
        try:
            self.content_page = requests.get(url)
            if not(self.content_page.status_code == requests.codes.ok):
                self.content_page.raise_for_status()
        except requests.exceptions.RequestException as ex:
            logging.info('A requests exception has ocurred: ' + str(ex))
            logging.error(traceback.format_exc())
            sys.exit(0)