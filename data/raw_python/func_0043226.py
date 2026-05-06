def migrate(action):
    " Migration utils [create, run, undo, redo]. "
    from flaskext.evolution import Evolution
    from flask import current_app
    evolution = Evolution(current_app)
    evolution.manager(action)