import numpy as np
import sciris as sc
import json
import jsbeautifier
from jsonobject import *
from jsonobject.base_properties import DefaultProperty
from jsonobject.containers import JsonDict
import os

from . import logger
from . import defaults
import warnings


class PopulationAgeDistribution(JsonObject):
    """Class for population age distribution with a specified number of bins."""
    num_bins = IntegerProperty()
    # [min_age, max_age, percentage]
    distribution = ListProperty(DefaultProperty)

class SexByAgeDistribution(JsonObject):
    """Class for sex by age distribution with a specified number of bins."""
    num_bins = IntegerProperty()
    # [min_age, max_age, percentagemale, percentagefemale]
    distribution = ListProperty(DefaultProperty)

class EducationByAgeDistribution(JsonObject):
    """Class for education by age distribution with a specified number of bins."""
    num_bins = IntegerProperty()
    # [min_age, max_age, percentageacrossgroups, noschooling, primary, lowersecondary, uppersecondary, postsecondarynontertiary, tertiary]
    maledistribution = ListProperty(DefaultProperty)
    femaledistribution = ListProperty(DefaultProperty)

class OccupationBySexDistribution(JsonObject):
    """Class for occupation by sex distribution with a specified number of bins."""
    num_bins = IntegerProperty()
    # [min_age, max_age, percentagemale, percentagefemale]
    distribution = ListProperty(DefaultProperty)

class OccupationByAgeDistribution(JsonObject):
    """Class for occupation by age distribution with a specified number of bins."""
    num_bins = IntegerProperty()
    # [min_age, max_age, percentagemale, percentagefemale]
    distribution = ListProperty(DefaultProperty)

class LongTermIllnessByAgeDistribution(JsonObject):
    """Class for long term illness by age distribution with a specified number of bins."""
    num_bins = IntegerProperty()
    # [min_age, max_age, percentagemale, percentagefemale]
    maledistribution = ListProperty(DefaultProperty)
    femaledistribution = ListProperty(DefaultProperty)

class BMIByAgeDistribution(JsonObject):
    """Class for BMI by age distribution with a specified number of bins."""
    num_bins = IntegerProperty()
    # [min_age, max_age, percentagemale, percentagefemale]
    maledistribution = ListProperty(DefaultProperty)
    femaledistribution = ListProperty(DefaultProperty)

class EmploymentStatusByAgeDistribution(JsonObject):
    """Class for Employment status by age distribution with a specified number of bins."""
    num_bins = IntegerProperty()
    # [min_age, max_age, percentageemployed, percentageunemployed, percentageinactive]
    maledistribution = ListProperty(DefaultProperty)
    femaledistribution = ListProperty(DefaultProperty)

class EmploymentByIndustryDistribution(JsonObject):
    """Class for Employment by Industry distribution with a specified number of bins."""
    num_bins = IntegerProperty()
    # [industry, male, female]
    maledistribution = ListProperty(DefaultProperty)
    femaledistribution = ListProperty(DefaultProperty)

class EmploymentByIndustryFtPtDistribution(JsonObject):
    """Class for Employment by Industry distribution with a specified number of bins."""
    num_bins = IntegerProperty()
    # [industry, male, female]
    maledistribution = ListProperty(DefaultProperty)
    femaledistribution = ListProperty(DefaultProperty)

class SchoolSizeDistributionByType(JsonObject):
    """Class for the school size distribution by school type."""
    school_type = StringProperty()
    # length should be len(location.school_size_distribution)
    size_distribution = ListProperty(DefaultProperty)

class SchoolTypeByAge(JsonObject):
    """Class for the school type by age range."""
    school_type = StringProperty()
    # [min_age, max_age]
    age_range = ListProperty(DefaultProperty)

class TourismDistributions(JsonObject):
    total_inbound_tourists = IntegerProperty()
    avg_inbound_tourists = DefaultProperty()
    total_inbound_nights = IntegerProperty()
    total_outbound_tourists = IntegerProperty()
    avg_outbound_tourists = DefaultProperty()
    total_outbound_nights = IntegerProperty()
    accommodation_capacity = ListProperty(IntegerProperty)
    gender = ListProperty(DefaultProperty)
    age = ListProperty(DefaultProperty)
    quarter = ListProperty(DefaultProperty)
    duration = ListProperty(DefaultProperty)
    accommodation_type = ListProperty(DefaultProperty)


class Location(JsonObject):
    """
    Class for the json object for the location containing data about the
    population to generate representative contact networks.

    The general use case of this is to use a filepath, and the parent data is
    parsed from the filepath. DefaultProperty type handles either a scalar or
    json object. We allow a json object mainly for testing of inheriting from a
    parent specified directly in the json.

    Most users will want to populate this with a relative or absolute file path.

    Note:
        The structures for the population age distribution will be updated to be
        more flexible to take in a parameter for the number of age brackets to
        generate the population age distribution structure.
    """
    location_name = StringProperty()
    data_provenance_notices = ListProperty(StringProperty)
    reference_links = ListProperty(StringProperty)
    citations = ListProperty(StringProperty)
    notes = ListProperty(StringProperty)

    parent = DefaultProperty()

    population_age_distributions = ListProperty(PopulationAgeDistribution)

    sex_byage_distributions = ListProperty(SexByAgeDistribution)

    education_byage_distributions = ListProperty(EducationByAgeDistribution)

    occupation_bysex_distributions = ListProperty(OccupationBySexDistribution)

    occupation_byage_distributions = ListProperty(OccupationByAgeDistribution)

    longtermillness_byage_distributions = ListProperty(LongTermIllnessByAgeDistribution)

    bmi_byage_distributions = ListProperty(BMIByAgeDistribution)

    employment_status_byage_distributions = ListProperty(EmploymentStatusByAgeDistribution)

    employment_byindustry_distributions = ListProperty(EmploymentByIndustryDistribution)

    employment_byindustry_ftpt_distributions = ListProperty(EmploymentByIndustryFtPtDistribution)

    numofcomorbidities_byage = ListProperty(
        # [minage, maxage, 1, 2, 3, 4, 5]
        ListProperty(DefaultProperty)
    )

    comorbiditytypes_byage = ListProperty(
        # [minage, maxage, 1, 2, 3, 4, 5]
        ListProperty(DefaultProperty)
    )

    employment_rates_by_age = ListProperty(
        # [age, percentage]
        ListProperty(DefaultProperty)
    )

    enrollment_rates_by_age = ListProperty(
        # [age, percentage]
        ListProperty(DefaultProperty)
    )

    household_head_age_brackets = ListProperty(
        # [age_min, age_max]
        ListProperty(DefaultProperty)
    )

    household_head_age_distribution_by_family_size = ListProperty(
        # length should be len(household_head_age_brackets) + 1
        # The first entry is the family size, the rest of the entries fill in the household head age counts for
        # each household head age bracket.
        # [family_size, count_1, count_2, ...]
        ListProperty(DefaultProperty)
    )

    household_size_distribution = ListProperty(
        # [size, percentage]
        ListProperty(DefaultProperty)
    )

    ltcf_resident_to_staff_ratio_distribution = ListProperty(
        # [ratio_low, ratio_hi, percentage]
        ListProperty(DefaultProperty)
    )

    ltcf_num_residents_distribution = ListProperty(
        # [num_residents_low, num_residents_hi, percentage]
        ListProperty(DefaultProperty)
    )

    ltcf_num_staff_distribution = ListProperty(
        # [num_staff_low, num_staff_hi, percentage]
        ListProperty(DefaultProperty)
    )

    ltcf_use_rate_distribution = ListProperty(
        # [age, percentage]
        ListProperty(DefaultProperty)
    )

    institutions_rates_by_age = ListProperty(
        # [minage, maxage, percentage]
        ListProperty(DefaultProperty)
    )

    institutions_type_distribution = ListProperty(
        # [type, percentage]
        ListProperty(DefaultProperty)
    )

    school_size_brackets = ListProperty(
        # [school_size_low, school_size_hi]
        ListProperty(DefaultProperty)
    )

    school_size_distribution = ListProperty(DefaultProperty)

    # The length of size_distribution needs to equal the length of school_size_brackets
    school_size_distribution_by_type = ListProperty(SchoolSizeDistributionByType)

    school_types_by_age = ListProperty(SchoolTypeByAge)

    workplace_size_counts_by_num_personnel = ListProperty(
        # [num_personnel_low, num_personnel_hi, count]
        ListProperty(DefaultProperty)
    )

    tourism_distributions = TourismDistributions()

    def get_list_properties(self):
        """
        Get the properties of the location data object as a list.

        Returns:
            list: A list of the properties of the location json object with
            data about the location.
        """
        return [p for p in self if type(getattr(self, p)) is JsonArray]

    def get_population_age_distribution(self, nbrackets):
        """
        Get the age distribution of the population aggregated to nbrackets age
        brackets. If the data doesn't contain a distribution with the requested number
        of brackets, an exception is raised.

        Args:
            nbrackets (int): the number of age brackets the age distribution is aggregated to

        Returns:
            list: A list of the probability age distribution values indexed by
            the bracket number.
        """

        matching_distributions = [d for d in self.population_age_distributions if d.num_bins==nbrackets]
        if len(matching_distributions) == 0:
            raise RuntimeError(f"The configured location data doesn't have a population age "
                               f"distribution with [{nbrackets}] brackets.")

        dist = matching_distributions[0].distribution
        return dist

    def get_sex_byage_distribution(self, nbrackets):
        """
        Get the sex by age group distribution of the population aggregated to nbrackets age
        brackets. If the data doesn't contain a distribution with the requested number
        of brackets, an exception is raised.

        Args:
            nbrackets (int): the number of age brackets the age distribution is aggregated to

        Returns:
            list: A list of the probability sex by age distribution values indexed by
            the bracket number.
        """

        matching_distributions = [d for d in self.sex_byage_distributions if d.num_bins==nbrackets]
        if len(matching_distributions) == 0:
            raise RuntimeError(f"The configured location data doesn't have a sex by age "
                               f"distribution with [{nbrackets}] brackets.")

        dist = matching_distributions[0].distribution
        return dist

    def get_education_byage_distribution(self, nbrackets):
        """
        Get the education by age group distribution of the population aggregated to nbrackets age
        brackets. If the data doesn't contain a distribution with the requested number
        of brackets, an exception is raised.

        Args:
            nbrackets (int): the number of age brackets the distribution is aggregated to

        Returns:
            list: A list of the probability education by age distribution values indexed by
            the bracket number.
        """

        matching_distributions = [d for d in self.education_byage_distributions if d.num_bins==nbrackets]
        if len(matching_distributions) == 0:
            raise RuntimeError(f"The configured location data doesn't have an education by age "
                               f"distribution with [{nbrackets}] brackets.")

        maledist = matching_distributions[0].maledistribution
        femaledist = matching_distributions[0].femaledistribution
        return maledist, femaledist

    def get_occupations_distributions(self, sex_nbrackets, age_nbrackets):
        """
        Get the occupation by sex/age group distribution of the population aggregated to nbrackets age
        brackets. If the data doesn't contain a distribution with the requested number
        of brackets, an exception is raised.

        Args:
            nbrackets (int): the number of brackets the distribution is aggregated to

        Returns:
            list: A list of the probability occupations by sex/ age distribution values indexed by
            the bracket number.
        """

        sex_matching_distributions = [d for d in self.occupation_bysex_distributions if d.num_bins==sex_nbrackets]
        if len(sex_matching_distributions) == 0:
            raise RuntimeError(f"The configured location data doesn't have an occupations by sex/age (male) "
                               f"distribution with [{sex_nbrackets}] brackets.")

        age_matching_distributions = [d for d in self.occupation_byage_distributions if d.num_bins==age_nbrackets]
        if len(age_matching_distributions) == 0:
            raise RuntimeError(f"The configured location data doesn't have an occupations by sex/age (female) "
                               f"distribution with [{age_nbrackets}] brackets.")

        return sex_matching_distributions[0].distribution, age_matching_distributions[0].distribution

    def get_longtermillness_byage_distribution(self, nbrackets):
        """
        Get the long term illness by age group distribution of the population aggregated to nbrackets age
        brackets. If the data doesn't contain a distribution with the requested number
        of brackets, an exception is raised.

        Args:
            nbrackets (int): the number of age brackets the distribution is aggregated to

        Returns:
            list: A list of the probability long term illness by age distribution values indexed by
            the bracket number.
        """

        matching_distributions = [d for d in self.longtermillness_byage_distributions if d.num_bins==nbrackets]
        if len(matching_distributions) == 0:
            raise RuntimeError(f"The configured location data doesn't have a long term illness by age "
                               f"distribution with [{nbrackets}] brackets.")

        maledist = matching_distributions[0].maledistribution
        femaledist = matching_distributions[0].femaledistribution
        return maledist, femaledist

    def get_numofcomorbidities_byage_distribution(self):
        """
        Get the num of comorbidities by age group distribution of the population
        brackets.

        Returns:
            list: A list of the probability num of comorbidities by age distribution values indexed by
            the bracket number.
        """

        matching_distributions = [d for d in self.numofcomorbidities_byage_distribution]
        if len(matching_distributions) == 0:
            raise RuntimeError(f"The configured location data doesn't have a num of comoridities by age "
                               f"distribution.")

        return matching_distributions

    def get_comorbiditytypes_byage_distribution(self):
        """
        Get the comorbidity types by age group distribution of the population
        brackets.

        Returns:
            list: A list of the probability cororbidity type by age distribution values indexed by
            the bracket number.
        """

        matching_distributions = [d for d in self.comorbiditytypes_byage_distribution]
        if len(matching_distributions) == 0:
            raise RuntimeError(f"The configured location data doesn't have a comorbidity types by age "
                               f"distribution.")

        return matching_distributions

    def get_bmi_byage_distribution(self, nbrackets):
        """
        Get the bmi by age group distribution of the population aggregated to nbrackets age
        brackets. If the data doesn't contain a distribution with the requested number
        of brackets, an exception is raised.

        Args:
            nbrackets (int): the number of age brackets the distribution is aggregated to

        Returns:
            list: A list of the probability bmi by age distribution values indexed by
            the bracket number.
        """

        matching_distributions = [d for d in self.bmi_byage_distributions if d.num_bins==nbrackets]
        if len(matching_distributions) == 0:
            raise RuntimeError(f"The configured location data doesn't have a BMI "
                               f"distribution with [{nbrackets}] brackets.")

        maledist = matching_distributions[0].maledistribution
        femaledist = matching_distributions[0].femaledistribution
        return maledist, femaledist

    def get_employmentstatus_byage_distribution(self, nbrackets):
        """
        Get the employment status by age group distribution of the population aggregated to nbrackets age
        brackets. If the data doesn't contain a distribution with the requested number
        of brackets, an exception is raised.

        Args:
            nbrackets (int): the number of age brackets the distribution is aggregated to

        Returns:
            list: A list of the probability employment status by age distribution values indexed by
            the bracket number.
        """

        matching_distributions = [d for d in self.employment_status_byage_distributions if d.num_bins==nbrackets]
        if len(matching_distributions) == 0:
            raise RuntimeError(f"The configured location data doesn't have a BMI "
                               f"distribution with [{nbrackets}] brackets.")

        maledist = matching_distributions[0].maledistribution
        femaledist = matching_distributions[0].femaledistribution
        return maledist, femaledist

    def get_employment_fulltime_parttime_distribution(self):
        """
        Get the employment fulltime vs parttime distribution.

        Returns:
            list: A list of the probability employment fulltime vs parttime
        """

        matching_distributions = [d for d in self.employment_fulltime_parttime_distributions]
        if len(matching_distributions) == 0:
            raise RuntimeError(f"The configured location data doesn't have an employment fulltime vs parttime "
                               f"distribution.")

        return matching_distributions

    def get_employment_byindustry_distribution(self, nbrackets):
        """
        Get the employment distribution of the population aggregated to nbrackets industry
        brackets. If the data doesn't contain a distribution with the requested number
        of brackets, an exception is raised.

        Args:
            nbrackets (int): the number of brackets the distribution is aggregated to

        Returns:
            list: A list of the probability employment by industry distribution values indexed by
            the bracket number.
        """

        matching_distributions = [d for d in self.employment_byindustry_distributions if d.num_bins==nbrackets]
        if len(matching_distributions) == 0:
            raise RuntimeError(f"The configured location data doesn't have an employment by industry "
                               f"distribution with [{nbrackets}] brackets.")

        matching_distributions_ftpt = [d for d in self.employment_byindustry_ftpt_distributions if d.num_bins==nbrackets]
        if len(matching_distributions) == 0:
            raise RuntimeError(f"The configured location data doesn't have an employment by industry (ftpt) "
                               f"distribution with [{nbrackets}] brackets.")

        maledist = matching_distributions[0].maledistribution
        femaledist = matching_distributions[0].femaledistribution
        maledistftpt = matching_distributions_ftpt[0].maledistribution
        femaledistftpt = matching_distributions_ftpt[0].femaledistribution
        return maledist, femaledist, maledistftpt, femaledistftpt

    def get_tourism_distributions(self):
        """
        Get the tourism distributions

        Args:
            nbrackets (int): the number of brackets the distribution is aggregated to

        Returns:
            TourismDistributions object
        """

        matching_distributions = self.tourism_distributions
        if matching_distributions is None:
            raise RuntimeError(f"The configured location data doesn't have any tourism distributions")

        return matching_distributions

def populate_parent_data_from_file_path(location, parent_file_path):
    """
    Loading a location json object with necessary data fields filled from the
    parent location using the parent location file path.

    Args:
        location (json)        : json object for the location data
        parent_file_path (str) : file path to the parent location

    Returns:
        json: The location json object with necessary data fields filled from
        the parent location.
    """
    # DM: parameter name of location should change to better reflect what this parameter actually is: the location data object
    logger.debug(f"Loading parent location from filepath [{parent_file_path}]")
    try:
        parent_obj = load_location_from_filepath(parent_file_path, check_constraints=False)
        location = populate_parent_data_from_json_obj(location, parent_obj)
    except:
        logger.warning(f"You may have an invalid data configuration: couldn't load parent "
                    f"from filepath [{parent_file_path}] for location [{location.location_name}]")
    return location


def populate_parent_data_from_json_obj(location, parent):
    """
    Loading a location json object with necessary data fields filled from the
    parent location json.

    Args:
        location (json) : json object for the location data
        parent (json)   : json object for the parent location

    Returns:
        json: The location json object with necessary data fields filled from
        the parent location.
    """
    # DM: parameter names should change to reflect that better
    if parent.parent is not None:
        populate_parent_data(parent)

    for list_property in location.get_list_properties():
        child_value = getattr(location, list_property)
        if len(child_value) == 0 and str(list_property) in parent:
            parent_value = parent[str(list_property)]
            if len(parent_value) > 0:
                setattr(location, list_property, parent_value)

    return location


def populate_parent_data(location):
    """
    Populate location json object with fields from the parent location if
    available.

    Args:
        location (json): json data object for the location  # parameter name change for more specificity

    Returns:
        json: The location json data object with data fields filled from the
        parent location.
    """
    if location.parent is None:
        return location

    parent = location.parent

    if type(parent) is str:
        if len(parent) == 0:
            return location
        return populate_parent_data_from_file_path(location, parent)

    if type(parent) is JsonDict:
        parent_location = Location(parent)
        return populate_parent_data_from_json_obj(location, parent_location)

    raise RuntimeError(f'Invalid type for parent field: [{type(parent)}]')


def load_location_from_json(json_obj, check_constraints=None):
    """
    Load location data from json object with some checks made.

    Args:
        json_obj (json): json object containing location data

    Returns:
        json: The json object with location data.
    """
    if check_constraints is None:
        check_constraints = True

    location = Location(json_obj)

    populate_parent_data(location)

    if check_constraints:
        check_location_constraints_satisfied(location)
        check_all_probability_distribution_sums(location)
        check_all_probability_distribution_nonnegative(location)

    return location


def load_location_from_json_str(json_str, check_constraints=None):
    """
    Load location data from json str with some checks made.

    Args:
        json_str (str): string version of the json object

    Returns:
        json: The json object with location data.
    """
    json_obj = json.loads(json_str)
    return load_location_from_json(json_obj, check_constraints=check_constraints)


def get_relative_path(datadir):
    """
    Get the relative path for the data folder.

    Args:
        datadir (str): data folder path

    Returns:
        str: Relative path for the data folder.

    Notes:
        This method may not be necessary anymore...
    """
    base_dir = datadir
    if len(defaults.settings.relative_path) > 1:
        base_dir = os.path.join(datadir,  *defaults.settings.relative_path)
    return base_dir


def get_location_attr(location, property_name):
    """
    Get the attribute from the json object containing location data given the
    associated property name.

    Args:
        location (json)     : the json object with location data
        property_name (str) : the property name

    Returns:
        If property_name exists in the location json object, return [True, attribute].
        Else, return [False, None].
    """
    if property_name in location.keys():
        return getattr(location, property_name)
    else:
        return [False, None]


def load_location_from_filepath(rel_filepath, check_constraints=None):
    """
    Loads location data object from provided relative filepath where the file path is
    relative to defaults.settings.datadir.

    Args:
        rel_filepath (str): relative file path for the location data

    Returns:
        json: The json object with location data.
    """
    if check_constraints is None:
        check_constraints = True

    filepath = os.path.join(get_relative_path(defaults.settings.datadir), rel_filepath)
    logger.debug(f"Opening location from filepath [{filepath}]")
    f = open(filepath, 'r')
    json_obj = json.load(f)
    return load_location_from_json(json_obj, check_constraints=check_constraints)


def save_location_to_filepath(location, abs_filepath):
    """
    Saves json object with location data to provided absolute filepath.

    Args:
        location (json)    : the json object with location data
        abs_filepath (str) : absolute file path to where the json is saved

    Returns:
        None.
    """
    logger.debug(f"Saving location json to filepath [{abs_filepath}]")
    location_json = location.to_json()

    options = jsbeautifier.default_options()
    options.indent_size = 2
    location_json = jsbeautifier.beautify(json.dumps(location_json), options)

    with open(abs_filepath, 'w') as f:
        f.write(location_json)
        # json.dump(location_json, f, indent=2)


def check_location_constraints_satisfied(location):
    """
    Checks a number of constraints that need to be satisfied for the schema.

    Args:
        location (json): the json object with location data

    Returns:
        None.

    Raises:
        RuntimeError with a description if one of the constraints is not
        satisfied.
    """
    [status, msg] = are_location_constraints_satisfied(location)
    if not status:
        raise RuntimeError(msg)


def are_location_constraints_satisfied(location):
    """
    Checks a number of constraints that need to be satisfied for the schema.

    Args:
        location (json): the json object with location data

    Returns:
        [True, None] if all constraints are satisfied.
        [False, str] if a constraint is violated. The returned str is one of
        the error messages.
    """

    for f in [check_location_name,
              check_population_age_distributions,
              check_employment_rates_by_age,
              check_enrollment_rates_by_age,
              check_household_head_age_brackets,
              check_household_head_age_distributions_by_family_size,
              check_household_size_distribution,
              check_ltcf_resident_to_staff_ratio_distribution,
              check_ltcf_num_residents_distribution,
              check_ltcf_num_staff_distribution,
              check_institutions_rates_by_age,
              check_institutions_type_distribution,
              check_school_size_brackets,
              check_school_size_distribution,
              check_school_size_distribution_by_type,
              check_school_types_by_age,
              check_workplace_size_counts_by_num_personnel,
              ]:
        [status, msg] = f(location)  # update this to return the combination of all the error messages
        if not status:
            return [status, msg]

    return [True, None]


def check_array_of_array_entry_lens_arr(array_of_arrays, expected_len):
    for [k, bracket] in enumerate(array_of_arrays):
        if not len(bracket) == expected_len:
            return [False,
                    f"Entry [{k}] has invalid length: [{len(bracket)}]; should be [{expected_len}]"]
    return [True, None]


def check_array_of_arrays_entry_lens(location, expected_len, property_name):
    """
    Check that each array in an array of arrays has the expected length.

    Args:
        location (json)     : the json object with location data
        expected_len (int)  : the expected length of each sub array
        property_name (str) : the property name

    Returns:
        [True, None] if sub array length checks pass.
        [False, str] if sub array length checks fail. The returned str is the
        error message.
    """
    arr = get_location_attr(location, property_name)
    status, reason = check_array_of_array_entry_lens_arr(arr, expected_len)
    if not status:
        return [False, f"For property {property_name}: {reason}"]

    return [True, None]


def check_valid_probability_distributions(property_name, valid_properties=None):
    """
    Check that the property_name is a valid probability distribution.

    Args:
        property_name (str)            : the property name
        valid_properties (str or list) : a list of the valid probability distributions

    Returns:
        None.
    """
    # check the property_name is in the list of valid_probability_distributions()
    if valid_properties is None:
        valid_properties = defaults.valid_probability_distributions

    # if a single str, make into a list so next check will work
    valid_properties = sc.tolist(valid_properties)

    if property_name not in valid_properties: # pragma: no cover
        raise NotImplementedError(f"{property_name} is not one of the expected probability distributions. The list of expected probability distributions is {valid_properties}. If you wish to use this method on the attribute {property_name}, you can supply it as the parameter valid_properties={property_name}.")

def check_probability_distribution_sum_age_distributions(location, arr, tolerance=1e-2, **kwargs):
    """
    Check that each population age distribution has a sum equal to 1 within some
    tolerance.

    Args:
        location (json)   : the json object with location data
        arr (list)        : the list of population age distributions
        tolerance (float) : difference from the sum of 1 tolerated
        kwargs (dict)     : dictionary of values passed to np.isclose()

    Returns:
        [True, None] if the sum of the probability distribution is equal to 1 within the tolerance level.
        [False, str] else. The returned str is the error message with some information about the check.
    """
    if tolerance is not None: # pragma: no cover
        kwargs['atol'] = tolerance

    checks, msgs = [], []
    for i in arr: # pragma: no cover
        if 'num_bins' in i:
            arr_i = np.array(i.distribution)
            arr_sum = np.sum(arr_i[:, -1])

            check = np.isclose(a=1, b=arr_sum, **kwargs)
            checks.append(check)

            if check:
                msg = ''
            else:
                msg = f"The sum of the probability distribution for the population age distribution for {location.location_name} with num_bins = {i.num_bins} is {arr_sum:.4f}.\n"
            msgs.append(msg)

        else:
            checks.append(False)
            msgs.append(f"The probability distribution for the population age distribution for {location.location_name} does not have num_bins.")
    msg = "".join(msgs)
    if msg == "": # pragma: no cover
        msg = None
    return [sum(checks) > 0, msg]

def check_probability_distribution_generic(location, arr, startfromindex, propertyname, tolerance=1e-2, **kwargs):
    """
    Check that each row has a sum equal to 1 within some tolerance. 
    Works with the typical distribution field and assumes the presence of 'num_bins'
    It also assumes zero or more precending fields (and no trailing fields)

    Args:
        location (json)   : the json object with location data
        arr (list)        : the list of distributions
        start from index  : if 2, ignore the first 2 elements in the row for e.g. could be minage, maxage
        propertyname      : property name (for logging purposes only)
        tolerance (float) : difference from the sum of 1 tolerated
        kwargs (dict)     : dictionary of values passed to np.isclose()

    Returns:
        [True, None] if the sum of the probability distribution is equal to 1 within the tolerance level.
        [False, str] else. The returned str is the error message with some information about the check.
    """
    if tolerance is not None: # pragma: no cover
        kwargs['atol'] = tolerance

    checks, msgs = [], []
    for i in arr: # pragma: no cover
        if 'num_bins' in i:
            arr_i = np.array(i.distribution)

            for x in arr_i:
                arr_i_x = np.array(x)

                arr_sum = np.sum(arr_i_x[startfromindex:])

                check = np.isclose(a=1, b=arr_sum, **kwargs)
                checks.append(check)

                if check:
                    msg = ''
                else:
                    msg = f"The sum of the probability distribution for the " + propertyname + " distribution for {location.location_name} with num_bins = {i.num_bins} is {arr_sum:.4f}.\n"
                msgs.append(msg)
        else:
            checks.append(False)
            msgs.append(f"The probability distribution for the " + propertyname + " distribution for {location.location_name} does not have num_bins.")
    msg = "".join(msgs)
    if msg == "": # pragma: no cover
        msg = None
    return [sum(checks) > 0, msg]


def check_probability_distribution_generic_nonumbins(location, arr, startfromindex, propertyname, tolerance=1e-2, **kwargs):
    """
    Check that each row has a sum equal to 1 within some tolerance. 
    Works without the typical distribution field (simple arrays)
    It also assumes zero or more precending fields (and no trailing fields)

    Args:
        location (json)   : the json object with location data
        arr (list)        : the list of distributions
        start from index  : if 2, ignore the first 2 elements in the row for e.g. could be minage, maxage
        propertyname      : property name (for logging purposes only)
        tolerance (float) : difference from the sum of 1 tolerated
        kwargs (dict)     : dictionary of values passed to np.isclose()

    Returns:
        [True, None] if the sum of the probability distribution is equal to 1 within the tolerance level.
        [False, str] else. The returned str is the error message with some information about the check.
    """
    if tolerance is not None: # pragma: no cover
        kwargs['atol'] = tolerance

    checks, msgs = [], []
    for x in arr: # pragma: no cover
        arr_i_x = np.array(x)

        arr_sum = np.sum(arr_i_x[startfromindex:])

        check = np.isclose(a=1, b=arr_sum, **kwargs)
        checks.append(check)

        if check:
            msg = ''
        else:
            msg = f"The sum of the probability distribution for the " + propertyname + " distribution for {location.location_name} with num_bins = {i.num_bins} is {arr_sum:.4f}.\n"
        msgs.append(msg)
        
    msg = "".join(msgs)
    if msg == "": # pragma: no cover
        msg = None
    return [sum(checks) > 0, msg]

def check_probability_distribution_sum_age_distributions(location, arr, tolerance=1e-2, **kwargs):
    """
    Check that each population age distribution has a sum equal to 1 within some
    tolerance.

    Args:
        location (json)   : the json object with location data
        arr (list)        : the list of population age distributions
        tolerance (float) : difference from the sum of 1 tolerated
        kwargs (dict)     : dictionary of values passed to np.isclose()

    Returns:
        [True, None] if the sum of the probability distribution is equal to 1 within the tolerance level.
        [False, str] else. The returned str is the error message with some information about the check.
    """
    if tolerance is not None: # pragma: no cover
        kwargs['atol'] = tolerance

    checks, msgs = [], []
    for i in arr: # pragma: no cover
        if 'num_bins' in i:
            arr_i = np.array(i.distribution)
            arr_sum = np.sum(arr_i[:, -1])

            check = np.isclose(a=1, b=arr_sum, **kwargs)
            checks.append(check)

            if check:
                msg = ''
            else:
                msg = f"The sum of the probability distribution for the population age distribution for {location.location_name} with num_bins = {i.num_bins} is {arr_sum:.4f}.\n"
            msgs.append(msg)

        else:
            checks.append(False)
            msgs.append(f"The probability distribution for the population age distribution for {location.location_name} does not have num_bins.")
    msg = "".join(msgs)
    if msg == "": # pragma: no cover
        msg = None
    return [sum(checks) > 0, msg]

def check_probability_distribution_sum_sex_byage_distributions(location, arr, tolerance=1e-2, **kwargs):
    """
    Check that each population age distribution has a sum equal to 1 within some
    tolerance.

    Args:
        location (json)   : the json object with location data
        arr (list)        : the list of population age distributions
        tolerance (float) : difference from the sum of 1 tolerated
        kwargs (dict)     : dictionary of values passed to np.isclose()

    Returns:
        [True, None] if the sum of the probability distribution is equal to 1 within the tolerance level.
        [False, str] else. The returned str is the error message with some information about the check.
    """
    if tolerance is not None: # pragma: no cover
        kwargs['atol'] = tolerance

    checks, msgs = [], []
    for i in arr: # pragma: no cover
        if 'num_bins' in i:
            arr_i = np.array(i.distribution)

            for x in arr_i:
                arr_i_x = np.array(x)

                arr_sum = np.sum(arr_i_x[2:])

                check = np.isclose(a=1, b=arr_sum, **kwargs)
                checks.append(check)

                if check:
                    msg = ''
                else:
                    msg = f"The sum of the probability distribution for the sex by age distribution for {location.location_name} with num_bins = {i.num_bins} is {arr_sum:.4f}.\n"
                msgs.append(msg)
        else:
            checks.append(False)
            msgs.append(f"The probability distribution for the sex by age distribution for {location.location_name} does not have num_bins.")
    msg = "".join(msgs)
    if msg == "": # pragma: no cover
        msg = None
    return [sum(checks) > 0, msg]

# def check_probability_distribution_sum_education_byage_distributions(location, arr, tolerance=1e-2, **kwargs):
#     """
#     Check that each education age distribution has a sum equal to 1 within some
#     tolerance.

#     Args:
#         location (json)   : the json object with location data
#         arr (list)        : the list of population age distributions
#         tolerance (float) : difference from the sum of 1 tolerated
#         kwargs (dict)     : dictionary of values passed to np.isclose()

#     Returns:
#         [True, None] if the sum of the probability distribution is equal to 1 within the tolerance level.
#         [False, str] else. The returned str is the error message with some information about the check.
#     """
#     if tolerance is not None: # pragma: no cover
#         kwargs['atol'] = tolerance

#     checks, msgs = [], []
#     for i in arr: # pragma: no cover
#         if 'num_bins' in i:
#             arr_i = np.array(i.maledistribution)

#             for x in arr_i:
#                 arr_i_x = np.array(x)

#                 arr_sum = np.sum(arr_i_x[2:]) #first cell minage, second cell maxage, third cell percentage across age groups, hence start from 3 onwards

#                 check = np.isclose(a=1, b=arr_sum, **kwargs)
#                 checks.append(check)

#                 if check:
#                     msg = ''
#                 else:
#                     msg = f"The sum of the probability distribution for the education by age distribution (male case) for {location.location_name} with num_bins = {i.num_bins} is {arr_sum:.4f}.\n"
#                 msgs.append(msg)

#             arr_i = np.array(i.femaledistribution)

#             for x in arr_i:
#                 arr_i_x = np.array(x)

#                 arr_sum = np.sum(arr_i_x[2:]) #first cell minage, second cell maxage, third cell percentage across age groups, hence start from 3 onwards

#                 check = np.isclose(a=1, b=arr_sum, **kwargs)
#                 checks.append(check)

#                 if check:
#                     msg = ''
#                 else:
#                     msg = f"The sum of the probability distribution for the education by age distribution (female case) for {location.location_name} with num_bins = {i.num_bins} is {arr_sum:.4f}.\n"
#                 msgs.append(msg)

#         else:
#             checks.append(False)
#             msgs.append(f"The probability distribution for the education by age distribution for {location.location_name} does not have num_bins.")

#     msg = "".join(msgs)
#     if msg == "": # pragma: no cover
#         msg = None
#     return [sum(checks) > 0, msg]

def check_probability_distribution_sum_male_female_byage_distributions(location, arr, property_name, start_index, tolerance=1e-2, **kwargs):
    """
    Check that each age distribution for both male and female has a sum equal to 1 within some
    tolerance.

    Args:
        location (json)   : the json object with location data
        arr (list)        : the list of population age distributions
        tolerance (float) : difference from the sum of 1 tolerated
        kwargs (dict)     : dictionary of values passed to np.isclose()

    Returns:
        [True, None] if the sum of the probability distribution is equal to 1 within the tolerance level.
        [False, str] else. The returned str is the error message with some information about the check.
    """
    if tolerance is not None: # pragma: no cover
        kwargs['atol'] = tolerance

    checks, msgs = [], []
    for i in arr: # pragma: no cover
        if 'num_bins' in i:
            arr_i = np.array(i.maledistribution)

            for x in arr_i:
                arr_i_x = np.array(x)

                arr_sum = np.sum(arr_i_x[start_index:]) #first cell minage, second cell maxage, third cell percentage across age groups, hence start from 3 onwards

                check = np.isclose(a=1, b=arr_sum, **kwargs)
                checks.append(check)

                if check:
                    msg = ''
                else:
                    msg = f"The sum of the probability distribution for the " + property_name + " distribution (male case) for {location.location_name} with num_bins = {i.num_bins} is {arr_sum:.4f}.\n"
                msgs.append(msg)

            arr_i = np.array(i.femaledistribution)

            for x in arr_i:
                arr_i_x = np.array(x)

                arr_sum = np.sum(arr_i_x[start_index:]) #first cell minage, second cell maxage, third cell percentage across age groups, hence start from 3 onwards

                check = np.isclose(a=1, b=arr_sum, **kwargs)
                checks.append(check)

                if check:
                    msg = ''
                else:
                    msg = f"The sum of the probability distribution for the " + property_name + " distribution (female case) for {location.location_name} with num_bins = {i.num_bins} is {arr_sum:.4f}.\n"
                msgs.append(msg)

        else:
            checks.append(False)
            msgs.append(f"The probability distribution for the " + property_name + " distribution for {location.location_name} does not have num_bins.")

    msg = "".join(msgs)
    if msg == "": # pragma: no cover
        msg = None
    return [sum(checks) > 0, msg]

def check_probability_distribution_sum_employment_distributions(location, arr, property_name, tolerance=1e-2, **kwargs):
    """
    Check that each distribution for employment distributions, have a sum equal to 1 within some
    tolerance.

    Args:
        location (json)   : the json object with location data
        arr (list)        : the list of population age distributions
        tolerance (float) : difference from the sum of 1 tolerated
        kwargs (dict)     : dictionary of values passed to np.isclose()

    Returns:
        [True, None] if the sum of the probability distribution is equal to 1 within the tolerance level.
        [False, str] else. The returned str is the error message with some information about the check.
    """
    if tolerance is not None: # pragma: no cover
        kwargs['atol'] = tolerance

    checks, msgs = [], []
    for i in arr: # pragma: no cover
        if 'num_bins' in i:
            arr_i = np.array(i.maledistribution)
            arr_sum = np.sum([x[1] for x in arr_i])
            check = np.isclose(a=1, b=arr_sum, **kwargs)
            checks.append(check)

            if check:
                msg = ''
            else:
                msg = f"The sum of the probability distribution for the " + property_name + " distribution (male) for {location.location_name} with num_bins = {i.num_bins} is {arr_sum:.4f}.\n"
            
            msgs.append(msg)


            arr_i = np.array(i.femaledistribution)
            arr_sum = np.sum([x[1] for x in arr_i])
            check = np.isclose(a=1, b=arr_sum, **kwargs)
            checks.append(check)

            if check:
                msg = ''
            else:
                msg = f"The sum of the probability distribution for the " + property_name + " distribution (male) for {location.location_name} with num_bins = {i.num_bins} is {arr_sum:.4f}.\n"
            
            msgs.append(msg)
        else:
            checks.append(False)
            msgs.append(f"The probability distribution for the " + property_name + " distribution for {location.location_name} does not have num_bins.")

    msg = "".join(msgs)
    if msg == "": # pragma: no cover
        msg = None
    return [sum(checks) > 0, msg]

def check_probability_distribution_sum_employment_ftpt_distributions(location, arr, property_name, tolerance=1e-2, **kwargs):
    """
    Check that each distribution for employment distributions, have a sum equal to 1 within some
    tolerance.

    Args:
        location (json)   : the json object with location data
        arr (list)        : the list of population age distributions
        tolerance (float) : difference from the sum of 1 tolerated
        kwargs (dict)     : dictionary of values passed to np.isclose()

    Returns:
        [True, None] if the sum of the probability distribution is equal to 1 within the tolerance level.
        [False, str] else. The returned str is the error message with some information about the check.
    """
    if tolerance is not None: # pragma: no cover
        kwargs['atol'] = tolerance

    checks, msgs = [], []
    for i in arr: # pragma: no cover
        if 'num_bins' in i:
            arr_i = np.array(i.maledistribution)

            for x in arr_i:
                arr_i_x = np.array(x)
                arr_sum = np.sum(arr_i_x[1:])
                check = np.isclose(a=1, b=arr_sum, **kwargs)
                checks.append(check)

                if check:
                    msg = ''
                else:
                    msg = f"The sum of the probability distribution for the " + property_name + " distribution (male) for {location.location_name} with num_bins = {i.num_bins} is {arr_sum:.4f}.\n"
                
                msgs.append(msg)

            arr_i = np.array(i.femaledistribution)

            for x in arr_i:
                arr_i_x = np.array(x)
                arr_sum = np.sum(arr_i_x[1:])
                check = np.isclose(a=1, b=arr_sum, **kwargs)
                checks.append(check)

                if check:
                    msg = ''
                else:
                    msg = f"The sum of the probability distribution for the " + property_name + " distribution (male) for {location.location_name} with num_bins = {i.num_bins} is {arr_sum:.4f}.\n"
                
                msgs.append(msg)
        else:
            checks.append(False)
            msgs.append(f"The probability distribution for the " + property_name + " distribution for {location.location_name} does not have num_bins.")

    msg = "".join(msgs)
    if msg == "": # pragma: no cover
        msg = None
    return [sum(checks) > 0, msg]

def check_probability_distribution_nonnegative_age_distributions(location, arr):
    """
    Check that each population age distribution has all non negative values.

    Args:
        location (json) : the json object with location data
        arr (list) : the list of population age distributions

    Returns:
        [True, None] if the sum of the probability distribution is equal to 1 within the tolerance level.
        [False, str] else. The returned str is the error message with some information about the check.
    """
    checks, msgs = [], []
    for i in arr: # pragma: no cover
        if 'num_bins' in i:
            arr_i = np.array(i.distribution)

            # find the indices where the distribution is negative
            negative = np.argwhere(arr_i < 0)
            # check is any are negative
            any_negative = len(negative)
            check = not any_negative
            checks.append(check)

            if check:
                msg = ''
            else:
                msg = f"The probability distribution for the population age distribution for {location.location_name} with num_bins = {i.num_bins} has some negative values, {arr_i[negative]}, at the indices {negative}.\n"
            msgs.append(msg)

        else:
            checks.append(False)
            msgs.append(f"The probability distribution for the population age distribution for {location.location_name} does not have num_bins.")
    msg = "".join(msgs)
    if msg == "": # pragma: no cover
        msg = None
    return [sum(checks) > 0, msg]

def check_probability_distribution_nonnegative_age_distributions_nonumbins(location, property_name, arr):
    """
    Check that each distribution has all non negative values.

    Args:
        location (json) : the json object with location data
        arr (list) : the list of population age distributions

    Returns:
        [True, None] if the sum of the probability distribution is equal to 1 within the tolerance level.
        [False, str] else. The returned str is the error message with some information about the check.
    """
    checks, msgs = [], []
    for i in arr: # pragma: no cover
        arr_i = np.array(i)

        # find the indices where the distribution is negative
        negative = np.argwhere(arr_i < 0)
        # check is any are negative
        any_negative = len(negative)
        check = not any_negative
        checks.append(check)

        if check:
            msg = ''
        else:
            msg = f"The probability distribution for the " + property_name + " distribution for {location.location_name} with num_bins = {i.num_bins} has some negative values, {arr_i[negative]}, at the indices {negative}.\n"
        msgs.append(msg)

    msg = "".join(msgs)
    if msg == "": # pragma: no cover
        msg = None
    return [sum(checks) > 0, msg]

def check_probability_distribution_nonnegative_malefemalesplit_distributions(location, arr):
    """
    Check that each population age distribution has all non negative values.

    Args:
        location (json) : the json object with location data
        arr (list) : the list of population age distributions

    Returns:
        [True, None] if the sum of the probability distribution is equal to 1 within the tolerance level.
        [False, str] else. The returned str is the error message with some information about the check.
    """
    checks, msgs = [], []
    for i in arr: # pragma: no cover
        if 'num_bins' in i:
            arr_i = np.array(i.maledistribution)

            # find the indices where the distribution is negative
            negative = np.argwhere(arr_i < 0)
            # check is any are negative
            any_negative = len(negative)
            check = not any_negative
            checks.append(check)

            if check:
                msg = ''
            else:
                msg = f"The probability distribution for the education age male distribution for {location.location_name} with num_bins = {i.num_bins} has some negative values, {arr_i[negative]}, at the indices {negative}.\n"
            msgs.append(msg)

            arr_i = np.array(i.femaledistribution)

            # find the indices where the distribution is negative
            negative = np.argwhere(arr_i < 0)
            # check is any are negative
            any_negative = len(negative)
            check = not any_negative
            checks.append(check)

            if check:
                msg = ''
            else:
                msg = f"The probability distribution for the education age female distribution for {location.location_name} with num_bins = {i.num_bins} has some negative values, {arr_i[negative]}, at the indices {negative}.\n"
            msgs.append(msg)

        else:
            checks.append(False)
            msgs.append(f"The probability distribution for the education age distribution for {location.location_name} does not have num_bins.")
    msg = "".join(msgs)
    if msg == "": # pragma: no cover
        msg = None
    return [sum(checks) > 0, msg]

def check_probability_distribution_nonnegative_employment_parttime_distributions(location, arr):
    """
    Check that each distribution has all non negative values.

    Args:
        location (json) : the json object with location data
        arr (list) : the list of distributions

    Returns:
        [True, None] if the sum of the probability distribution is equal to 1 within the tolerance level.
        [False, str] else. The returned str is the error message with some information about the check.
    """
    checks, msgs = [], []
    for i in arr: # pragma: no cover
        if 'num_bins' in i:
            arr_i = np.array(i.ptwithftdistribution)

            # find the indices where the distribution is negative
            negative = np.argwhere(arr_i < 0)
            # check is any are negative
            any_negative = len(negative)
            check = not any_negative
            checks.append(check)

            if check:
                msg = ''
            else:
                msg = f"The probability distribution for the employment (part time with full time) distribution for {location.location_name} with num_bins = {i.num_bins} has some negative values, {arr_i[negative]}, at the indices {negative}.\n"
            msgs.append(msg)

            arr_i = np.array(i.ptwithoutftdistribution)

            # find the indices where the distribution is negative
            negative = np.argwhere(arr_i < 0)
            # check is any are negative
            any_negative = len(negative)
            check = not any_negative
            checks.append(check)

            if check:
                msg = ''
            else:
                msg = f"The probability distribution for the employment (part timewithout full time) distribution for {location.location_name} with num_bins = {i.num_bins} has some negative values, {arr_i[negative]}, at the indices {negative}.\n"
            msgs.append(msg)

        else:
            checks.append(False)
            msgs.append(f"The probability distribution for the employment (part time) distribution for {location.location_name} does not have num_bins.")
    msg = "".join(msgs)
    if msg == "": # pragma: no cover
        msg = None
    return [sum(checks) > 0, msg]


def check_probability_distribution_sum(location, property_name, tolerance=1e-2, valid_properties=None, **kwargs):
    """
    Check that fields representing probability distributions have sums equal to 1 within some tolerance.

    Args:
        location (json)                : the json object with location data
        property_name (str)            : the property name
        tolerance (float)              : difference from the sum of 1 tolerated
        valid_properties (str or list) : a list of the valid probability distributions
        kwargs (dict)                  : dictionary of values passed to np.isclose()

    Returns:
        [True, None] if the sum of the probability distribution is equal to 1 within the tolerance level.
        [False, str] else. The returned str is the error message with some information about the check.
    """
    check_valid_probability_distributions(property_name, valid_properties)

    # is the absolute difference between the sum and the expected value of 1 less than the tolerance value?
    if tolerance is not None:
        kwargs['atol'] = tolerance

    arr = get_location_attr(location, property_name)

    if property_name == 'population_age_distributions':
        check, msg = check_probability_distribution_sum_age_distributions(location, arr, **kwargs)
        return check, msg
    elif property_name == 'sex_byage_distributions' or property_name == 'occupation_bysex_distributions' or property_name == 'occupation_byage_distributions':
        startfromindex = 0

        if property_name == 'sex_byage_distributions' or property_name == 'occupation_byage_distributions':
            startfromindex = 2
        else:
            startfromindex = 1

        check, msg = check_probability_distribution_generic(location, arr, startfromindex, property_name, **kwargs)
        return check, msg
    elif property_name == "employment_byindustry_distributions":
        check, msg = check_probability_distribution_sum_employment_distributions(location, arr, property_name,  **kwargs)
        return check, msg
    elif property_name == "employment_byindustry_ftpt_distributions":
        check, msg = check_probability_distribution_sum_employment_ftpt_distributions(location, arr, property_name,  **kwargs)
        return check, msg
    # elif property_name == "education_byage_distributions":
    #     check, msg = check_probability_distribution_sum_education_byage_distributions(location, arr, **kwargs)
    #     return check, msg
    elif property_name == "education_byage_distributions" or property_name == "longtermillness_byage_distributions" or property_name == "bmi_byage_distributions" or property_name == "employment_status_byage_distributions":
        start_index = 2

        check, msg = check_probability_distribution_sum_male_female_byage_distributions(location, arr, property_name, start_index, **kwargs)
        return check, msg
    elif property_name == "numofcomorbidities_byage_distribution" or property_name == "comorbiditytypes_byage_distribution":
        startfromindex = 2

        check, msg = check_probability_distribution_generic_nonumbins(location, arr, startfromindex, property_name, **kwargs)
        return check, msg
    elif len(arr):

        arr = np.array(arr)

        if arr.ndim == 1:  # for school size distributions
            arr_sum = sum(arr)  # what is the sum of the probability distribution values?

        elif arr.ndim == 2:
            arr_sum = np.sum(arr[:, -1])  # distribution values are in the last column if arr is 2D array

        else:
            raise NotImplementedError(f"Could not understand an array of shape {arr.shape}: Expected a 1D or 2D array.")

        check = np.isclose(a=1, b=arr_sum, **kwargs)

        if check:
            return [True, None]
        else:
            return [False, f"The sum of the probability distribution for the property: {property_name} is {arr_sum:.4f}.\n\
We expected the sum of these probabilities to be less than {tolerance} from 1."]
    else:
        return [False, f"{location.location_name} {property_name} could not be checked for a sum close to 1."]


def check_probability_distribution_nonnegative(location, property_name, valid_properties=None):
    """
    Check that fields representing probability distributions have all non negative values.

    Args:
        location (json)                : the json object with location data
        property_name (str)            : the property name
        valid_properties (str or list) : a list of the valid probability distributions

    Returns:
        [True, None] if the values of the probability distribution are all non negative.
        [False, str] else. The returned str is the error message with some information about the check.
    """
    check_valid_probability_distributions(property_name, valid_properties)

    arr = get_location_attr(location, property_name)

    if property_name == 'population_age_distributions' or property_name == 'sex_byage_distributions' or property_name == 'occupation_bysex_distributions' or property_name == 'occupation_byage_distributions':
        check, msg = check_probability_distribution_nonnegative_age_distributions(location, arr)
        return check, msg
    elif property_name == 'education_byage_distributions' or property_name == "longtermillness_byage_distributions" or property_name == "bmi_byage_distributions" or property_name == "employment_status_byage_distributions" or property_name == "employment_byindustry_distributions" or property_name == 'employment_byindustry_ftpt_distributions':
        check, msg = check_probability_distribution_nonnegative_malefemalesplit_distributions(location, arr)
        return check, msg
    elif property_name == "numofcomorbidities_byage_distribution" or property_name == "comorbiditytypes_byage_distribution":
        check, msg = check_probability_distribution_nonnegative_age_distributions_nonumbins(location, property_name, arr)
        return check, msg
    elif len(arr):
        arr = np.array(arr)

        if arr.ndim == 2:
            arr = arr[:, -1]  # distribution values are in the last column if arr is 2D array

        # find the indices where the distribution is negative
        negative = np.argwhere(arr < 0)
        # check if any are negative
        any_negative = len(negative)
        check = not any_negative

        if check:
            return [True, None]
        else:
            return [False, f"The probability distribution for the property: {property_name} has some negative values, {arr[negative]}, at the indices {negative}."]
    else:
        return [False, f"{location.location_name} {property_name} could not be checked for negative values."]


def check_all_probability_distribution_sums(location, tolerance=1e-2, die=False, verbose=False, **kwargs):
    """
    Checks that each probability distribution available to a location has a sum
    close to 1.

    Args:
        location (json)   : the json object with location data
        tolerance (float) : difference from the sum of 1 tolerated
        die (bool)        : raise an exception if the check fails
        verbose (bool)    : print a warning if the check fails
        kwargs (dict)     : dictionary of values passed to np.isclose()

    Returns:
        list, list: List of checks and a list of associated error messages.
    """
    property_list = defaults.valid_probability_distributions

    checks, msgs = [], []

    for i, property_name in enumerate(property_list):
        check, msg = check_probability_distribution_sum(location, property_name, tolerance=tolerance, **kwargs)
        checks.append(check)
        msgs.append(msg)

        if not check:
            if die: # pragma: no cover
                raise ValueError(msg)
            elif verbose:
                warnings.warn(msg)
        logger.debug(f"Check passed. The sum of the probability distribution for {property_name} is within {tolerance} of 1. ")
    return checks, msgs


def check_all_probability_distribution_nonnegative(location, die=False, verbose=True):
    """
    Run checks that a field representing probabilty distributions has all non
    negative values.

    Args:
        location (json) : json object with the location data
        die (bool)      : raise an exception if the check fails
        verbose (bool)  : print a warning if the check fails

    Returns:
        list, list: List of checks and a list of associated error messages.
    """
    property_list = defaults.valid_probability_distributions

    checks, msgs = [], []

    for i, property_name in enumerate(property_list):
        check, msg = check_probability_distribution_nonnegative(location, property_name)
        checks.append(check)
        msgs.append(msg)

        if not check:
            if die: # pragma: no cover
                raise ValueError(msg)
            elif verbose:
                warnings.warn(msg)
        logger.debug(f"Check passed. The probability distribution for {property_name} has all non negative values.")
    return checks, msgs


def check_location_name(location):
    """
    Check the location json data object has a string.

    Args:
        location (json): the json object with location data

    Returns:
        [True, str] if the location json has a str value in the location_name
        field. Returned str specifies the location_name.
        [False, str] if the location json does not have a str value in the
        location_name field.
    """
    if location.location_name is not None and len(location.location_name) > 0 and isinstance(location.location_name, str):
        return [True, f"The location_name is {location.location_name}"]

    return [False, "location_name must be specified"]


def check_population_age_distributions(location):
    """
    Check that the population age distributions are self-consistent in the number of brackets,
    and each sub array has length 3.

    Args:
        location (json): the json object with location data

    Returns:
        [True, None] if checks pass. [False, str] if checks fail.
    """
    for population_age_distribution in location.population_age_distributions:
        if len(population_age_distribution.distribution) != population_age_distribution.num_bins:
            return [False, f"Length for {population_age_distribution} distribution doesn't match 'num_bins': "
                           f"{len(population_age_distribution.distribution)} != {population_age_distribution.num_bins}"]
        return check_array_of_array_entry_lens_arr(population_age_distribution.distribution, 3)
    return [True, None]


def check_employment_rates_by_age(location):
    """
    Check that the employment rates by age is an array of arrays, where each
    sub array has length 2.

    Args:
        location (json): the json object with location data

    Returns:
        [True, None] if checks pass. [False, str] if checks fail.
    """
    return check_array_of_arrays_entry_lens(location, 2, 'employment_rates_by_age')


def check_enrollment_rates_by_age(location):
    """
    Check that the enrollment rates by age is an array of arrays, where each
    sub array has length 2.

    Args:
        location (json): the json object with location data

    Returns:
        [True, None] if checks pass. [False, str] if checks fail.
    """
    return check_array_of_arrays_entry_lens(location, 2, 'enrollment_rates_by_age')


def check_household_head_age_brackets(location):
    """
    Check that the household head age brackets is an array of arrays, where each
    sub array has length 2.

    Args:
        location (json): the json object with location data

    Returns:
        [True, None] if checks pass. [False, str] if checks fail.
    """
    return check_array_of_arrays_entry_lens(location, 2, 'household_head_age_brackets')


def check_household_head_age_distributions_by_family_size(location):
    """
    Check that the conditional household head age distribution by household size
    is an array with length equal to the number of household head age brackets.

    Args:
        location (json): the json object with location data

    Returns:
        [True, None] if checks pass. [False, str] if checks fail.
    """
    num_household_age_brackets = len(location.household_head_age_brackets)

    for [k, household_head_age_distribution] in enumerate(location.household_head_age_distribution_by_family_size):
        expected_len = 1 + num_household_age_brackets
        actual_len = len(household_head_age_distribution)
        if not actual_len == expected_len:
            return [False,
                    f"Entry [{k}] in household_head_age_distribution_by_family_size has invalid length: [{actual_len}]; should be [{expected_len}]"]
    return [True, None]


def check_household_size_distribution(location):
    """
    Check that the household size distribution is an array of arrays, where each
    sub array has length 2.

    Args:
        location (json): the json object location data

    Returns:
        [True, None] if checks pass. [False, str] if checks fail.
    """
    return check_array_of_arrays_entry_lens(location, 2, 'household_size_distribution')

def check_ltcf_resident_to_staff_ratio_distribution(location):
    """
    Check that the long term care facility resident to staff ratio distribution
    is an array of arrays, where each sub array has length 3.

    Args:
        location (json): the json object location data

    Returns:
        [True, None] if checks pass. [False, str] if checks fail.
    """
    return check_array_of_arrays_entry_lens(location, 3, 'ltcf_resident_to_staff_ratio_distribution')


def check_ltcf_num_residents_distribution(location):
    """
    Check that the long term care facility resident size distribution
    is an array of arrays, where each sub array has length 3.

    Args:
        location (json): the json object location data

    Returns:
        [True, None] if checks pass. [False, str] if checks fail.
    """
    return check_array_of_arrays_entry_lens(location, 3, 'ltcf_num_residents_distribution')


def check_ltcf_num_staff_distribution(location):
    """
    Check that the long term care facility staff size distribution is an array
    of arrays, where each sub array has length 3.

    Args:
        location (json): the json object location data

    Returns:
        [True, None] if checks pass. [False, str] if checks fail.
    """
    return check_array_of_arrays_entry_lens(location, 3, 'ltcf_num_staff_distribution')

def check_institutions_rates_by_age(location):
    """
    Check that the institutions rates by age is an array of arrays, where each
    sub array has length 3.

    Args:
        location (json): the json object with location data

    Returns:
        [True, None] if checks pass. [False, str] if checks fail.
    """
    return check_array_of_arrays_entry_lens(location, 3, 'institutions_rates_by_age')


def check_institutions_type_distribution(location):
    """
    Check that the institutions type distribution is an array of arrays, where each
    sub array has length 2.

    Args:
        location (json): the json object with location data

    Returns:
        [True, None] if checks pass. [False, str] if checks fail.
    """
    return check_array_of_arrays_entry_lens(location, 2, 'institutions_type_distribution')


def check_school_size_brackets(location):
    """
    Check that the school size distribution brackets is an array of arrays,
    where each sub array has length 2.

    Args:
        location (json): the json object location data

    Returns:
        [True, None] if checks pass. [False, str] if checks fail.
    """
    return check_array_of_arrays_entry_lens(location, 2, 'school_size_brackets')


def check_school_size_distribution(location):
    # TODO: decide if there is a check we should apply here.
    # DM: This should check that the school size distribution has the same
    # length as the school size brackets otherwise we have a data inconsistency
    return [True, None]


def check_school_size_distribution_by_type(location):
    """
    Check that the school size distribution by school type is an array of
    arrays, where each sub array has length 3.

    Args:
        location (json): the json object location data

    Returns:
        [True, None] if checks pass. [False, str] if checks fail.
    """
    num_school_size_brackets = len(location.school_size_brackets)

    for [k, bracket] in enumerate(location.school_size_distribution_by_type):
        expected_len = num_school_size_brackets
        actual_len = len(bracket.size_distribution)
        if not actual_len == num_school_size_brackets:
            return [False,
                    f"Entry [{k} - {bracket.school_type}] in school_size_distribution_by_type has invalid length for size_distribution: [{actual_len}]; should be [{expected_len}]"]
    return [True, None]


def check_school_types_by_age(location):
    """
    Check that the school types by age range is an array of arrays, where each
    sub array has length 2.

    Args:
        location (json): the json object location data

    Returns:
        [True, None] if checks pass. [False, str] if checks fail.
    """
    for [k, bracket] in enumerate(location.school_types_by_age):
        expected_len = 2
        actual_len = len(bracket.age_range)
        if not actual_len == expected_len:
            return [False,
                    f"Entry [{k} - {bracket.school_type}] in school_types_by_age has invalid length for age_range: [{actual_len}]; should be [{expected_len}]"]
    return [True, None]


def check_workplace_size_counts_by_num_personnel(location):
    """
    Check that the workplace size count is an array of arrays, where each sub
    array has length 3.

    Args:
        location (json): the json object location data

    Returns:
        [True, None] if checks pass. [False, str] if checks fail.
    """
    return check_array_of_arrays_entry_lens(location, 3, 'workplace_size_counts_by_num_personnel')


def convert_df_to_json_array(df, cols, int_cols=None):
    """
    Convert desired data from a pandas dataframe into a json array.

    Args:
        df (pandas dataframe)  : the dataframe with data
        cols (list)            : list of the columns to convert to the json array format
        int_cols (str or list) : a str or list of columns to convert to integer values

    Returns:
        array: An array version of the pandas dataframe to be added to synthpops
        json data objects.
    """
    df = df[cols]

    # make into a list to iterate over
    int_cols = sc.tolist(int_cols)

    # some columns as ints
    df = df.astype({k: int for k in int_cols})

    # make an array of arrays --- dtype=object to preserve each columns type
    arr = df.to_numpy(dtype=object).tolist()

    return arr
