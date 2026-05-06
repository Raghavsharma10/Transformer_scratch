def addParam(self, key, value, attrib=None):
        """ Checks the key dosnt already exist, adds alternate names to a seperate list

        Future
            - format input and add units
            - logging
        """

        if key in self.rejectTags:
            return False  # TODO Replace with exception

        # Temporary code to handle the seperation tag than can occur several times with different units.
        # TODO code a full multi unit solution (github issue #1)
        if key == 'separation':
            if attrib is None:
                return False  # reject seperations without a unit
            try:
                if not attrib['unit'] == 'AU':
                    return False  # reject for now
            except KeyError:  # a seperation attribute exists but not one for units
                return False

        if key in self.params:  # if already exists

            if key == 'name':
                try:  # if flagged as a primary or popular name use this one, an option should be made to use either
                    if attrib['type'] == 'pri':  # first names or popular names.
                        oldname = self.params['name']
                        self.params['altnames'].append(oldname)
                        self.params['name'] = value
                    else:
                        self.params['altnames'].append(value)
                except (KeyError, TypeError):  # KeyError = no type key in attrib dict, TypeError = not a dict
                    self.params['altnames'].append(value)
            elif key == 'list':
                self.params['list'].append(value)
            else:
                try:
                    name = self.params['name']
                except KeyError:
                    name = 'Unnamed'
                print('rejected duplicate {0}: {1} in {2}'.format(key, value, name))  # TODO: log rejected value
                return False  # TODO Replace with exception

        else:  # If the key doesn't already exist and isn't rejected

            # Some tags have no value but a upperlimit in the attributes
            if value is None and attrib is not None:
                try:
                    value = attrib['upperlimit']
                except KeyError:
                    try:
                        value = attrib['lowerlimit']
                    except KeyError:
                        return False

            if key == 'rightascension':
                value = _ra_string_to_unit(value)
            elif key == 'declination':
                value = _dec_string_to_unit(value)
            elif key in self._defaultUnits:
                try:
                    value = float(value) * self._defaultUnits[key]
                except:
                    print('caught an error with {0} - {1}'.format(key, value))
            self.params[key] = value