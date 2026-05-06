def parse_parameters(payflowpro_response_data):
    """
    Parses a set of Payflow Pro response parameter name and value pairs into 
    a list of PayflowProObjects, and returns a tuple containing the object
    list and a dictionary containing any unconsumed data. 
    
    The first item in the object list will always be the Response object, and
    the RecurringPayments object (if any) will be last.

    The presence of any unconsumed data in the resulting dictionary probably
    indicates an error or oversight in the PayflowProObject definitions.
    """
    def build_class(klass, unconsumed_data):
        known_att_names_set = set(klass.base_fields.keys())
        available_atts_set = known_att_names_set.intersection(unconsumed_data)
        if available_atts_set:
            available_atts = dict()
            for name in available_atts_set:
                available_atts[name] = unconsumed_data[name]
                del unconsumed_data[name]                    
            return klass(**available_atts)
        return None

    unconsumed_data = payflowpro_response_data.copy()

    # Parse the response data first
    response = build_class(Response, unconsumed_data)
    result_objects = [response]
    
    # Parse the remaining data
    for klass in object.__class__.__subclasses__(PayflowProObject):
        obj = build_class(klass, unconsumed_data)
        if obj:
            result_objects.append(obj)
    
    # Special handling of RecurringPayments
    p_count = 1
    payments = []
    while ("p_result%d" % p_count) in unconsumed_data:
        payments.append(RecurringPayment(
            p_result = unconsumed_data.pop("p_result%d" % p_count, None),
            p_pnref = unconsumed_data.pop("p_pnref%d" % p_count, None),
            p_transtate = unconsumed_data.pop("p_transtate%d" % p_count, None),
            p_tender = unconsumed_data.pop("p_tender%d" % p_count, None),
            p_transtime = unconsumed_data.pop("p_transtime%d" % p_count, None),
            p_amt = unconsumed_data.pop("p_amt%d" % p_count, None)))
        p_count += 1
    if payments:
        result_objects.append(RecurringPayments(payments=payments))
        
    return (result_objects, unconsumed_data,)