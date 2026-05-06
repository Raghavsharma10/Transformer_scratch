def makeMany2ManyRelatedManager(formodel, name_relmodel, name_formodel):
    '''formodel is the model which the manager .'''

    class _Many2ManyRelatedManager(Many2ManyRelatedManager):
        pass

    _Many2ManyRelatedManager.formodel = formodel
    _Many2ManyRelatedManager.name_relmodel = name_relmodel
    _Many2ManyRelatedManager.name_formodel = name_formodel
    return _Many2ManyRelatedManager