def checkAndRaise(pageNum, itemsPerPage):
        """Check and Raise an Exception if needed
        
        Args:
            pageNum (int): Page number
            itemsPerPage (int): Number of items per Page
            
        Raises:
            ErrPaginationLimits: If we are out of limits
        
        """
        if pageNum < 1:
            raise ErrPaginationLimits(ErrPaginationLimits.ERR_PAGE_NUM)
        
        if itemsPerPage < Settings.itemsPerPageMin or itemsPerPage > Settings.itemsPerPageMax:
            raise ErrPaginationLimits(ErrPaginationLimits.ERR_ITEMS_PER_PAGE)