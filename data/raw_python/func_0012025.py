def DeSelectByIndex(cls, index):
        ''' 通过索引，取消选择下拉框选项，
        @param index: 下拉框  索引
        '''
        try:
            Select(cls._element()).deselect_by_index(int(index))
        except:
            return False