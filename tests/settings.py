import sciris as sc

pop_sizes = sc.objdict(
    small        = 1e3,
    small_medium = 5e3,
    medium       = 8e3,
    medium_large = 12e3,
    large        = 20e3,
    malta        = 519562,
    ten_k        = 10e3,
    hundred_k    = 100e3
)

student_teacher_ratio = sc.objdict(
    small = 7.5, # 1k
    ten_k = 11, # 10k
    hundred_k = 10.5, # 100k
    malta = 10 # 519562
)

student_allstaff_ratio = sc.objdict(
    small = 3.9, # 1k
    ten_k = 5.3, # 10k
    hundred_k = 5.25, # 100l
    malta = 5.2 # 519562
)

total_inbound_tourists_override = sc.objdict(
    small = 4000,
    ten_k = 40000,
    hundred_k = 400000,
    malta = None
)

def get_full_feature_pars():
    pars = dict(
        n                               = pop_sizes.small_medium,
        rand_seed                       = 123,
        max_contacts                    = None,

        with_industry_code              = 0,
        with_facilities                 = 1,
        with_non_teaching_staff         = 1,
        use_two_group_reduction         = 1,
        with_school_types               = 1,

        average_LTCF_degree             = 20,
        ltcf_staff_age_min              = 20,
        ltcf_staff_age_max              = 60,

        school_mixing_type              = 'age_and_class_clustered',
        average_class_size              = 20,
        inter_grade_mixing              = 0.1,
        teacher_age_min                 = 25,
        teacher_age_max                 = 75,
        staff_age_min                   = 20,
        staff_age_max                   = 75,

        average_student_teacher_ratio   = 20,
        average_teacher_teacher_degree  = 3,
        average_student_all_staff_ratio = 15,
        average_additional_staff_degree = 20,
    )
    return pars