def add_error_handlers(app):
    """Add custom error handlers for PyMacaronCoreExceptions to the app"""

    def handle_validation_error(error):
        response = jsonify({'message': str(error)})
        response.status_code = error.status_code
        return response

    app.errorhandler(ValidationError)(handle_validation_error)