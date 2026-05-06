def verify(self):
        """Raise an |ValueError| if the dates or the step size of the time
        frame are inconsistent.
        """
        if self.firstdate >= self.lastdate:
            raise ValueError(
                f'Unplausible timegrid. The first given date '
                f'{self.firstdate}, the second given date is {self.lastdate}.')
        if (self.lastdate-self.firstdate) % self.stepsize:
            raise ValueError(
                f'Unplausible timegrid. The period span between the given '
                f'dates {self.firstdate} and {self.lastdate} is not '
                f'a multiple of the given step size {self.stepsize}.')