def load_json(json_object):
   ''' Load json from file or file name '''
   content = None
   if isinstance(json_object, str) and os.path.exists(json_object):
      with open_(json_object) as f:
         try:
            content = json.load(f)
         except Exception as e:
            debug.log("Warning: Content of '%s' file is not json."%f.name)
   elif hasattr(json_object, 'read'):
      try:
         content = json.load(json_object)
      except Exception as e:
         debug.log("Warning: Content of '%s' file is not json."%json_object.name)
   else:
      debug.log("%s\nWarning: Object type invalid!"%json_object)
   return content