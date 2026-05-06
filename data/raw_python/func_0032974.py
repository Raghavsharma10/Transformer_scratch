def run(self):
        """ Construct the document id from the date and the url. """
        document = {}
        document['_id'] = hashlib.sha1('%s:%s' % (
                                       self.date, self.url)).hexdigest()
        with self.input().open() as handle:
            document['content'] = handle.read().decode('utf-8', 'ignore')
        document['url'] = self.url
        document['date'] = unicode(self.date)
        with self.output().open('w') as output:
            output.write(json.dumps(document))