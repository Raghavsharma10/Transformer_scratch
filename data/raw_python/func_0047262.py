def uuid4(self):
    """Make an id in the format of UUID4, but keep in mind this could very well be pseudorandom, and if it is you'll not be truely random, and can regenerate same id if same seed"""
    return ''.join([hexchars[self.randint(0,15)] for x in range(0,8)]) + '-' +\
           ''.join([hexchars[self.randint(0,15)] for x in range(0,4)]) + '-' +\
           '4'+''.join([hexchars[self.randint(0,15)] for x in range(0,3)]) + '-' +\
           uuid4special[self.randint(0,3)]+''.join([hexchars[self.randint(0,15)] for x in range(0,3)]) + '-' +\
           ''.join([hexchars[self.randint(0,15)] for x in range(0,12)])