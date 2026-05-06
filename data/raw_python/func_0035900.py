def pad(data, length):
        """This function returns a padded version of the input data to the
        given length.  this function will shorten the given data to the length
        specified if necessary.  post-condition: len(data) = length

        :param data: the data byte array to pad
        :param length: the length to pad the array to
        """
        if (len(data) > length):
            return data[0:length]
        else:
            return data + b"\0" * (length - len(data))