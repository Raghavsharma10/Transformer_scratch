def choice(self,arr):
    """Uniform random selection of a member of an list

    :param arr: list you want to select an element from
    :type arr: list
    :return: one element from the list
    """
    ind = self.randint(0,len(arr)-1)
    return arr[ind]