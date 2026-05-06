def directory_listing_content_check(directory_path, listing):
        """
        Checks if a given listing is present under the given directory path.

        :param directory_path: The path to the base directory
        :param listing: The listing to check
        :return: None if no errors could be found, otherwise a string describing the error
        """
        if listing:
            for sub in listing:
                path = os.path.join(directory_path, sub['basename'])
                if sub['class'] == 'File':
                    if not os.path.isfile(path):
                        return 'listing contains "{}" but this file could not be found on disk.'.format(path)
                elif sub['class'] == 'Directory':
                    if not os.path.isdir(path):
                        return 'listing contains "{}" but this directory could not be found on disk'.format(path)
                    listing = sub.get('listing')
                    if listing:
                        return ConnectorManager.directory_listing_content_check(path, listing)
        return None