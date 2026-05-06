def readSettings(settings_path=None):
    global _settings
    '''
    Reads the settings corresponding to the plugin from where the method is called.
    This function has to be called in the __init__ method of the plugin class.
    Settings are stored in a settings.json file in the plugin folder.
    Here is an eample of such a file:

    [
    {"name":"mysetting",
     "label": "My setting",
     "description": "A setting to customize my plugin",
     "type": "string",
     "default": "dummy string",
     "group": "Group 1"
     "onEdit": "def f():\\n\\tprint "Value edited in settings dialog"
     "onChange": "def f():\\n\\tprint "New settings value has been saved"
    },
    {"name":"anothersetting",
      "label": "Another setting",
     "description": "Another setting to customize my plugin",
     "type": "number",
     "default": 0,
     "group": "Group 2"
    },
    {"name":"achoicesetting",
     "label": "A choice setting",
     "description": "A setting to select from a set of possible options",
     "type": "choice",
     "default": "option 1",
     "options":["option 1", "option 2", "option 3"],
     "group": "Group 2"
    }
    ]

    Available types for settings are: string, bool, number, choice, crs and text (a multiline string)

    The onEdit property contains a function that will be executed when the user edits the value
    in the settings dialog. It shouldl return false if, after it has been executed, the setting
    should not be modified and should recover its original value.

    The onEdit property contains a function that will be executed when the setting is changed after
    closing the settings dialog, or programatically by callin the setPluginSetting method

    Both onEdit and onChange are optional properties

    '''

    namespace = _callerName().split(".")[0]
    settings_path = settings_path or os.path.join(os.path.dirname(_callerPath()), "settings.json")
    with open(settings_path) as f:
        _settings[namespace] = json.load(f)