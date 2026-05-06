def _conv_which_data(which_data):
        """Convert which data to string or tuple

        This function improves user convenience,
        as `which_data` may be of several types
        (str, ,str with spaces and commas, list, tuple) which
        is internally handled by this method.
        """
        if isinstance(which_data, str):
            which_data = which_data.lower().strip()
            if which_data.count(","):
                # convert comma string to list
                which_data = [w.strip() for w in which_data.split(",")]
                # remove empty strings
                which_data = [w for w in which_data if w]
                if len(which_data) == 1:
                    return which_data[0]
                else:
                    # convert to tuple
                    return tuple(which_data)
            else:
                return which_data
        elif isinstance(which_data, (list, tuple)):
            which_data = [w.lower().strip() for w in which_data]
            return tuple(which_data)
        elif which_data is None:
            return None
        else:
            msg = "unknown type for `which_data`: {}".format(which_data)
            raise ValueError(msg)