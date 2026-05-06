def clear_i2b2_tables(tables: I2B2Tables, uploadid: int) -> None:
    """
    Remove all entries in the i2b2 tables for uploadid.
    :param tables:
    :param uploadid:
    :return:
    """
    # This is a static function to support the removefacts operation
    print("Deleted {} patient_dimension records"
          .format(PatientDimension.delete_upload_id(tables, uploadid)))
    print("Deleted {} patient_mapping records"
          .format(PatientMapping.delete_upload_id(tables, uploadid)))
    print("Deleted {} observation_fact records"
          .format(ObservationFact.delete_upload_id(tables, uploadid)))
    print("Deleted {} visit_dimension records"
          .format(VisitDimension.delete_upload_id(tables, uploadid)))
    print("Deleted {} encounter_mapping records"
          .format(EncounterMapping.delete_upload_id(tables, uploadid)))