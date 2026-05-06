def randomCommaSeparatedIntegerField(self):
        """
        Return the unique integers in the string such as below:
            '6,1,7' or '4,5,1,3,2' or '2,7,9,3,5,4,1' or '3,9,2,8,7,1,5,4,6'
        """
        randint = lambda max: ",".join(
            [str(x) for x in random.sample(range(1, 10), max)]
        )
        lst = [
            randint(3),
            randint(5),
            randint(7),
            randint(9)
        ]
        return self.randomize(lst)