def get_collection_instance(klass, api_client = None, request_api=True, **kwargs):
    """
    instatiates the collection lookup of json type klass
    :param klass: json file name
    :param api_client: transportation api
    :param request_api: if True uses the default APIClient
    """
    _type = klass
    if api_client is None and request_api:
        api_client = api.APIClient()
    if isinstance(klass, dict):
        _type = klass['type']
    obj = CollectionResource(_type, api_client, **kwargs)
    return obj        
 
#
#    /**
#     * magic method for mapping all kinds of method calls to addFilter
#     * @param string $method method name
#     * @param array $args array of arguments
#     * @return SaleskingCollection
#     * @throws BadMethodCallException
#     * @since 1.0.0
#     */
#    public function __call($method, array $args) {
#        try {
#            $this->addFilter($method,$args[0]);
#            return $this;
#        }
#        catch (SaleskingException $e)
#        {
#            if($e->getCode() == "FILTER_NOTEXISTING")
#            {
#                throw new BadMethodCallException('Call to undefined method :'.$method);
#            }
#
#            throw $e;
#        }
#    }

    def sort(self, direction = "ASC"):
        """
        set the sort to the query
        ['ASC','DESC']
        """
        direction = directtion.upper()
        if direction in ['ASC','DESC']:
            self.sort = direction
        else:
            raise SaleskingException("SORT_INVALIDDIRECTION","Invalid sorting direction - please choose either ASC or DESC");
    
    def sort_by(self, property):
        """
        set sort by property to the query
        """
        seek =u"sort_by"
        # make sure that the api supports sorting for this kind of object
        if seek in self.schema['links']['instances']['properties']:
            #  make sure that we have a valid property
            if seek in self.schema['links']['instances']['properties']['sort_by']['enum']:
                self.sort_by = property
                return self
            else:
                raise SaleskingException("SORTBY_INVALIDPROPERTY","Invalid property for sorting");
        else:
            raise SaleskingException("SORTBY_CANNOTSORT","object type doesnt support sorting");