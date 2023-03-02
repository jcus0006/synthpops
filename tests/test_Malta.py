"""Test Malta location works and plot the demographics and contact networks."""
import sciris as sc
import synthpops as sp
import numpy as np
import matplotlib as mplt
import matplotlib.pyplot as plt
import settings
import pytest


pars = sc.objdict(
    n                =  settings.pop_sizes.small,
    rand_seed        =  123,

    household_method =  'fixed_ages',
    smooth_ages      =  1,

    country_location =  'Malta',
    sheet_name       =  'Malta',
    with_school_types = True,
    school_mixing_type = 'age_and_class_clustered',
    with_non_teaching_staff = True,
    average_student_teacher_ratio = settings.student_teacher_ratio.small, # 10 worked best based on Population of circa 500k, 7.5 worked best based on Population of 1k. Population data (end of 2020) and Enrollment data (2021) (Eurostat suggested a ratio of 8.8)
    average_student_all_staff_ratio = settings.student_allstaff_ratio.small, # 5.2 for 500k case, 3.9 for 1k case. average_student_teacher_ratio * 0.52. according to NSO
    teacher_age_min = 15, # min and max ages are not being restricted. because otherwise some workers from the distributions will not be assigned a workplace
    teacher_age_max = 95,
    staff_age_min = 15,
    staff_age_max = 95,
    with_facilities = True,
    ltcf_staff_age_min = 15,
    ltcf_staff_age_max = 95,
    tourism = True,
    use_default      =  True,
    save_to_json_file     = True
)


def test_Malta():
    """Test Malta population constructed."""
    sp.logger.info("Test that Malta contact networks can be made. Not a guarantee that the population made matches age mixing patterns well yet.")

    # reset the default location to pull other data
    sp.set_location_defaults(country_location="Senegal")
    # make a basic population
    pop = sp.Pop(**pars)
    assert pop.country_location == 'Malta', "population location information is not set to Malta"
    sp.reset_default_settings()  # reset defaults


def pop_exploration():
    sp.logger.info("Exploration of the Malta population generation with default methods and missing data filled in with Senegal data")
    sp.set_location_defaults(country_location="Senegal")
    pop = sp.Pop(**pars)
    print(pop.summarize())
    pop.plot_ages()
    pop.plot_enrollment_rates_by_age()
    sp.set_location_defaults()
    plt.show()
    sp.reset_default_settings()  # reset defaults


if __name__ == '__main__':
    test_Malta()
    pop_exploration()
