def save_training_samples(self, domain='', filename=''):
        """ Saves data previously added via add_training_sample().
            Data saved in folder specified by Train.get_corpus_path().

            :param domain: Name for domain folder.
                           If not set, current timestamp will be used.
            :param filename: Name for file to save data in.
                             If not set, file.txt will be used.

            Check the README file for more information about Domains.
        """
        self.trainer.save(domain=domain, filename=filename)