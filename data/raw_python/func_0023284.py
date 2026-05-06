def get_recipe_intent_handler(request):
    """
    You can insert arbitrary business logic code here    
    """

    # Get variables like userId, slots, intent name etc from the 'Request' object
    ingredient = request.slots["Ingredient"]  # Gets an Ingredient Slot from the Request object.
    
    if ingredient == None:
        return alexa.create_response("Could not find an ingredient!")

    # All manipulations to the request's session object are automatically reflected in the request returned to Amazon.
    # For e.g. This statement adds a new session attribute (automatically returned with the response) storing the
    # Last seen ingredient value in the 'last_ingredient' key. 

    request.session['last_ingredient'] = ingredient # Automatically returned as a sessionAttribute
    
    # Modifying state like this saves us from explicitly having to return Session objects after every response

    # alexa can also build cards which can be sent as part of the response
    card = alexa.create_card(title="GetRecipeIntent activated", subtitle=None,
                             content="asked alexa to find a recipe using {}".format(ingredient))    

    return alexa.create_response("Finding a recipe with the ingredient {}".format(ingredient),
                                 end_session=False, card_obj=card)