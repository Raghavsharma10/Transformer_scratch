def output_for_debugging(self, stream, data):
        """It will write the translated version of the file"""
        with open('%s.spec.out' % stream.name, 'w') as f: f.write(str(data))