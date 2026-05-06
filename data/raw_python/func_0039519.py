def isValidExp(self, exp):
        '''     
            Method to verify if a given expression is correct just in case the used regular expression needs additional processing to verify this fact.$
            This method will be overwritten when necessary.

            :param exp:     Expression to verify.

            :return:        True | False
        '''
        # order of the letters depending on which is the mod of the number
        #         0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15   16   17   18   19   20   21   22   23
        order = ['T', 'R', 'W', 'A', 'G', 'M', 'Y', 'F', 'P', 'D', 'X', 'B', 'N', 'J', 'Z', 'S', 'Q', 'V', 'H', 'L', 'C', 'K', 'E', 'T']

        #print exp
        l = exp[len(exp)-1]

        try:
            # verifying if it is an 8-length number
            number = int(exp[0:7])
        except:
            try:
                # verifying if it is a 7-length number
                number = int(exp[0:6])
            except:
                # not a  valid number
                pass
        if l == order[number%23]:
                    return True
        else:
            return False