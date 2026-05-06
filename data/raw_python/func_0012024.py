def SelectByIndex(cls, index):
        ''' 通过索引，选择下拉框选项，
        @param index: 下拉框  索引
        '''
        try:
            Select(cls._element()).select_by_index(int(index))
        except:
            return False