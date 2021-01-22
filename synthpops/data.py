import json
from jsonobject import *


class Location(JsonObject):
    data_provenance_notices = ListProperty(StringProperty)
    reference_links = ListProperty(StringProperty)
    citations = ListProperty(StringProperty)
    population_age_distribution = ListProperty(ListProperty(FloatProperty))
    employment_rates_by_age = ListProperty(ListProperty(FloatProperty))
    enrollment_rates_by_age = ListProperty(ListProperty(FloatProperty))
    household_head_age_brackets = ListProperty(ListProperty(FloatProperty))
    household_head_age_distribution_by_family_size = ListProperty(ListProperty(FloatProperty))
    household_size_distribution = ListProperty(ListProperty(FloatProperty))
    ltcf_resident_to_staff_ratio_distribution = ListProperty(ListProperty(FloatProperty))


def load_location_from_json(json_obj):
    location = Location(json_obj)
    return location


def load_location_from_json_str(json_str):
    json_obj = json.loads(json_str)
    return load_location_from_json(json_obj)


def load_location_from_filepath(filepath):
    f = open(filepath, 'r')
    json_obj = json.load(f)
    return load_location_from_json(json_obj)

