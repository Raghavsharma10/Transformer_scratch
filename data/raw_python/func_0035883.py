def _parseSpecType(self, classString):
        """ This class attempts to parse the spectral type. It should probably use more advanced matching use regex
        """

        try:
            classString = str(classString)
        except UnicodeEncodeError:
            # This is for the benefit of 1RXS1609 which currently has the spectral type K7\pm 1V
            # TODO add unicode support and handling for this case / ammend the target
            return False

        # some initial cases
        if classString == '' or classString == 'nan':
            return False

        possNumbers = range(10)
        possLType = ('III', 'II', 'Iab', 'Ia0', 'Ia', 'Ib', 'IV', 'V')  # in order of unique matches

        # remove spaces, remove slashes
        classString = classString.replace(' ', '')

        classString = classString.replace('-', '/')
        classString = classString.replace('\\', '/')
        classString = classString.split('/')[0]  # TODO we do not consider slashed classes yet (intemediates)

        # check first 3 chars for spectral types
        stellarClass = classString[:3]
        if stellarClass in _possSpectralClasses:
            self.classLetter = stellarClass
        elif stellarClass[:2] in _possSpectralClasses:  # needed because A5V wouldnt match before
            self.classLetter = stellarClass[:2]
        elif stellarClass[0] in _possSpectralClasses:
            self.classLetter = stellarClass[0]
        else:
            return False  # assume a non standard class and fail

        # get number
        try:
            numIndex = len(self.classLetter)
            classNum = int(classString[numIndex])
            if classNum in possNumbers:
                self.classNumber = int(classNum)  # don't consider decimals here, done at the type check
                typeString = classString[numIndex+1:]
            else:
                return False  # invalid number received
        except IndexError:  # reached the end of the string
            return True
        except ValueError:  # i.e its a letter - fail # TODO multi letter checking
            typeString = classString[1:]

        if typeString == '':  # ie there is no more information as in 'A8'
            return True

        # Now check for a decimal and handle those cases
        if typeString[0] == '.':
            # handle decimal cases, we check each number in turn, add them as strings and then convert to float and add
            # to original number
            decimalNumbers = '.'
            for number in typeString[1:]:
                try:
                    if int(number) in possNumbers:
                        decimalNumbers += number
                    else:
                        print('Something went wrong in decimal checking') # TODO replace with logging
                        return False # somethings gone wrong
                except ValueError:
                    break  # recevied a non-number (probably L class)
            #  add decimal to classNum
            try:
                self.classNumber += float(decimalNumbers)
            except ValueError: # probably trying to convert '.' to a float
                pass
            typeString = typeString[len(decimalNumbers):]
            if len(typeString) is 0:
                return True

        # Handle luminosity class
        for possL in possLType:  # match each possible case in turn (in order of uniqueness)
            Lcase = typeString[:len(possL)]  # match from front with length to minimise matching say IV in '<3 CIV'
            if possL == Lcase:
                self.lumType = possL
                return True

        if not self.classNumber == '':
            return True
        else:  # if there no number asumme we have a name ie 'Catac. var.'
            self.classLetter = ''
            self.classNumber = ''
            self.lumType = ''
            return False