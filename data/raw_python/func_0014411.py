def _gen_file_name(self):
        '''
        Generates a random file name based on self._output_filename_pattern for the output to do file.
        '''
        date = datetime.datetime.now()
        dt = "{}-{}-{}-{}-{}-{}-{}".format(str(date.year),str(date.month),str(date.day),str(date.hour),str(date.minute),str(date.second),str(random.randint(0,10000)))
        return self._output_filename_pattern.format(dt)