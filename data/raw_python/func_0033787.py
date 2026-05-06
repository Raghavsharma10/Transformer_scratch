def add_maxjobs_category(self,categoryName,maxJobsNum):
    """
    Add a category to this DAG called categoryName with a maxjobs of maxJobsNum.
    @param node: Add (categoryName,maxJobsNum) tuple to CondorDAG.__maxjobs_categories.
    """
    self.__maxjobs_categories.append((str(categoryName),str(maxJobsNum)))