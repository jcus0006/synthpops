import numpy as np
import random
import scipy.stats as stats
import matplotlib.pyplot as plt
import math

def generate_tourism(inbound_aggregates, outbound_aggregates, accom_capacities, group_size_dist, gender_dist, age_groups_dist, quarter_dist, duration_dist, accom_type_dist, purpose_dist, visualise = False):
    # accommodations = generate_accommodation(accom_capacities, visualise)

    # tourists = generate_inbound_tourists(inbound_aggregates["total_inbound_tourists"], gender_dist, age_groups_dist, quarter_dist, duration_dist, purpose_dist, accom_type_dist, visualise)

    group_sizes = generate_group_sizes(group_size_dist, inbound_aggregates["total_inbound_tourists"], visualise)

    return group_sizes
    # return accommodations

def generate_accommodation(accom_capacities, visualise = False):
    accomodations = {}

    for index, accom in enumerate(accom_capacities): # 1: hotel, 2: hostel/guest house, 3: tourist village, 4: rental accommodation
        # accommodation id
        accom_id = accom[0]

        # total number of units
        total_units = accom[1]

        # total number of beds across all units
        total_beds = accom[2]

        # minimum and maximum number of beds
        min_beds = accom[3]
        max_beds = accom[4]

        if total_units == 1:
            bed_counts = [total_beds]
        else:
            # shape parameter for gamma distribution
            gamma_shape = 1

            # Generate bed counts using rejection sampling (reject any samples not within range)
            bed_counts = []
            while len(bed_counts) < total_units:
                # Generate a gamma-distributed sample
                sample = sample_gamma(gamma_shape, min_beds, max_beds)
                # Check if the sample falls within the specified range
                if min_beds <= sample <= max_beds:
                    bed_counts.append(sample)

            total_generated_beds = sum(bed_counts)

            # Adjust the bed counts to ensure that the total number of beds is closer to the desired value
            for i in range(len(bed_counts)):
                temp_new_bed_count = round(bed_counts[i] * (total_beds / total_generated_beds))

                if min_beds <= temp_new_bed_count <= max_beds:
                    bed_counts[i] = temp_new_bed_count
                else:
                    if temp_new_bed_count < min_beds:
                        bed_counts[i] = min_beds
                    elif temp_new_bed_count > max_beds:
                        bed_counts[i] = max_beds

            total_adjusted_beds = sum(bed_counts)

            extra_beds = total_beds - total_adjusted_beds

            to_add = 0
            if extra_beds > 0:
                to_add = 1
            elif extra_beds < 0:
                to_add = -1

            if extra_beds != 0: # we need to either remove or add single beds until we get the required sum (check ranges again to never go out of range)
                bed_counts_len = len(bed_counts)

                while sum(bed_counts) != total_beds:
                    for i in range(bed_counts_len):
                        temp_new_bed_count = bed_counts[i] + to_add
                        
                        if min_beds <= temp_new_bed_count <= max_beds:
                            bed_counts[i] = temp_new_bed_count

                        if sum(bed_counts) == total_beds:
                            break              
                            
        if visualise:
            plt.figure(index+1)
            plt.hist(bed_counts, bins=30)
            plt.xlabel("Total number of beds")
            plt.ylabel("Frequency")
            plt.title("Distribution of number of beds")

            plt.show(block=False)

        # create a dictionary of hotel IDs and their corresponding bed counts
        accom_beds = {i: bed_counts[i] for i in range(total_units)}

        # print out the dictionary of hotel IDs and their corresponding bed counts
        accomodations[accom_id] = accom_beds

    return accomodations

def generate_inbound_tourists(num_inbound_tourists, gender_dist, age_groups_dist, quarter_dist, duration_dist, purpose_dist, accom_type_dist, visualise = False):
    tourists = {}

    gender_options = [index for index, g in enumerate(gender_dist)]

    age_ranges_options = [index for index, a in enumerate(age_groups_dist)]
    age_ranges_gamma = [(age_groups_dist[0][0], age_groups_dist[0][1], age_groups_dist[0][3]), (age_groups_dist[1][0], age_groups_dist[1][1], age_groups_dist[1][3]), (age_groups_dist[2][0], age_groups_dist[2][1], age_groups_dist[2][3]), (age_groups_dist[3][0], age_groups_dist[3][1], age_groups_dist[3][3])]
    age_groups_flat_dist = [age_groups_dist[0][2], age_groups_dist[1][2], age_groups_dist[2][2], age_groups_dist[3][2]] # [(0,24), (25,44), (45,64), (65,100)]

    quarter_ranges_options = [index for index, q in enumerate(quarter_dist)]
    quarter_ranges = [(quarter_dist[0][0], quarter_dist[0][1]), (quarter_dist[1][0], quarter_dist[1][1]), (quarter_dist[2][0], quarter_dist[2][1]), (quarter_dist[3][0], quarter_dist[3][1])]
    quarter_flat_dist = [quarter_dist[0][2], quarter_dist[1][2], quarter_dist[2][2], quarter_dist[3][2]] # [q1, q2, q3, q4]

    duration_ranges_options = [index for index, d in enumerate(duration_dist)]
    duration_ranges_gamma = [(duration_dist[0][0], duration_dist[0][1], duration_dist[0][3]), (duration_dist[1][0], duration_dist[1][1], duration_dist[1][3]), (duration_dist[2][0], duration_dist[2][1], duration_dist[2][3])]
    duration_flat_dist = [duration_dist[0][2], duration_dist[1][2], duration_dist[2][2]] # [(1,3), (4,6), (7,30)]

    purpose_options = [index for index, p in enumerate(purpose_dist)]
    purpose_flat_dist = [purpose_dist[0][1], purpose_dist[1][1], purpose_dist[2][1], purpose_dist[3][1]] # [hol, business, familyvisit, other]

    accom_type_options = [index for index, ac in enumerate(accom_type_dist)]
    accom_type_flat_dist = [accom_type_dist[0][1], accom_type_dist[1][1], accom_type_dist[2][1]] # [collective, rented, non-rented]

    for i in range(num_inbound_tourists):
        tourists[i] = None

        gender = np.random.choice(gender_options, size=len(gender_options), p=gender_dist)[0]
        # gender = random.choices(range(len(gender_dist)), weights=gender_dist)[0]

        age_range = np.random.choice(age_ranges_options, size=len(age_ranges_options), p=age_groups_flat_dist)[0]
        # age_range = random.choices(range(len(age_groups_flat_dist)), weights=age_groups_flat_dist)[0] # to apply Gamma distribution on lowest & largest age range, & uniform on the rest

        # age_gamma_shape = 1
        age_gamma_shape = age_ranges_gamma[age_range][2]

        if age_gamma_shape == 0: # use uniform distribution if 0, or gamma distribution if not
            age = random.randint(age_ranges_gamma[age_range][0], age_ranges_gamma[age_range][1])
        else:
            age = sample_gamma_reject_out_of_range(age_gamma_shape, age_ranges_gamma[age_range][0], age_ranges_gamma[age_range][1], 1, True, True)

        quarter_range = np.random.choice(quarter_ranges_options, size=len(quarter_ranges_options), p=quarter_flat_dist)[0]
        # quarter_range = random.choices(range(len(quarter_flat_dist)), weights=quarter_flat_dist)[0]

        month = random.randint(quarter_ranges[quarter_range][0], quarter_ranges[quarter_range][1])

        duration_range = np.random.choice(duration_ranges_options, size=len(duration_ranges_options), p=duration_flat_dist)[0]
        # duration_range = random.choices(range(len(duration_flat_dist)), weights=duration_flat_dist)[0] # to apply Gamma distribution on longest duration range, & uniform on the rest

        duration_gamma_shape = duration_ranges_gamma[duration_range][2]
        # duration_gamma_shape = 1

        if duration_gamma_shape == 0: # use uniform distribution if 0, or gamma distribution if not
            duration = random.randint(duration_ranges_gamma[duration_range][0], duration_ranges_gamma[duration_range][1]) 
        else:
            duration = sample_gamma_reject_out_of_range(duration_gamma_shape, duration_ranges_gamma[duration_range][0], duration_ranges_gamma[duration_range][1], 1, True, True)

        purpose = np.random.choice(purpose_options, size=len(purpose_options), p=purpose_flat_dist)[0]
        # purpose = random.choices(range(len(purpose_flat_dist)), weights=purpose_flat_dist)[0]

        accom_type = np.random.choice(accom_type_options, size=len(accom_type_options), p=accom_type_flat_dist)[0]
        # accom_type = random.choices(range(len(accom_type_flat_dist)), weights=accom_type_flat_dist)[0]

        tourists[i] = { "age": age, "gender": gender, "month": month, "duration": duration, "purpose": purpose, "accom_type": accom_type}

    if visualise:
        age_range_1 = [tourist["age"] for tourist in tourists.values() if tourist["age"] <= 24]
        age_range_4 = [tourist["age"] for tourist in tourists.values() if tourist["age"] >= 65]
        duration_range_3 = [tourist["duration"] for tourist in tourists.values() if tourist["duration"] >= 7]

        plt.figure(5)
        plt.hist(age_range_1, bins=100)
        plt.xlabel("Ages")
        plt.ylabel("Frequency")
        plt.title("Distribution of Age Range (0-24)")
        plt.show(block=False)

        plt.figure(6)
        plt.hist(age_range_4, bins=100)
        plt.xlabel("Ages")
        plt.ylabel("Frequency")
        plt.title("Distribution of Age Range (65-100)")
        plt.show(block=False)

        plt.figure(7)
        plt.hist(duration_range_3, bins=100)
        plt.xlabel("Days")
        plt.ylabel("Frequency")
        plt.title("Distribution of Duration Range (7-30)")
        plt.show(block=False)

    return tourists

def generate_group_sizes(group_size_dist, num_inbound_tourists, visualise=False):
    ranges = [(group_size_dist[0][0], group_size_dist[0][1]), (group_size_dist[1][0], group_size_dist[1][1]), (group_size_dist[2][0], group_size_dist[2][1]), (group_size_dist[3][0], group_size_dist[3][1])]
    percentages = [group_size_dist[0][2], group_size_dist[1][2], group_size_dist[2][2], group_size_dist[3][2]] # [(1,1), (2,2), (3,5), (6,10)]

    group_sizes = []
    for range_, percent in zip(ranges, percentages):
        if range_[0] == range_[1]:
            # if the range is just a single value, add it to the group sizes
            group_sizes += [range_[0]] * math.floor(percent * num_inbound_tourists)
        else:
            # if the range has multiple values, weight the lower end more
            min_range = range_[0]
            max_range = range_[1]

            size_range = range(min_range, max_range + 1) # size range e.g. range(3, 6)
            total_percent = sum(1/i for i in size_range) # sum of the inverse of each group size. e.g. 1/3 + 1/4 + 1/5 = 0.78
            weights =  [1 / (size * total_percent) for size in size_range] # normalize e.g. 1 / (3 * 0.78), 1 / (4 * 0.78). 1 / (5 * 0.78)
            
            # for each "size" in size range (e.g. 3, 4, 5), create "weight * percent * num_inbound_toursts" indices and assign "size" into each index
            sizes = [size for size in size_range for _ in range(math.floor(weights[size-range_[0]] * percent * num_inbound_tourists))]
            group_sizes += sizes # add to original group_sizes array

    if visualise:
        plt.figure(8)
        plt.hist(group_sizes, bins=10)
        plt.xlabel("Group sizes")
        plt.ylabel("Frequency")
        plt.title("Distribution of Group Sizes")
        plt.show(block=False)

    return group_sizes

    # for group_size_range in group_size_ranges:
    #     min_size = group_size_range[0]
    #     max_size = group_size_range[1]

    #     percentage = group_size_flat_dist[group_size_range]

    #     for size in range(min_size, max_size):


def generate_group_sizes2(group_size_dist, num_inbound_tourists, visualise=False):
    group_size_ranges = [(group_size_dist[0][0], group_size_dist[0][1]), (group_size_dist[1][0], group_size_dist[1][1]), (group_size_dist[2][0], group_size_dist[2][1]), (group_size_dist[3][0], group_size_dist[3][1])]
    group_size_flat_dist = [group_size_dist[0][2], group_size_dist[1][2], group_size_dist[2][2], group_size_dist[3][2]] # [(1,1), (2,2), (3,5), (6,10)]

    group_sizes = []

    while sum(group_sizes) < num_inbound_tourists:
        # sample new size
        group_size_range = random.choices(range(len(group_size_flat_dist)), weights=group_size_flat_dist)[0]

        min_val = group_size_ranges[group_size_range][0]
        max_val = group_size_ranges[group_size_range][1]

        group_size = 0
        if min_val == max_val:
            group_size = min_val
        else:
            gamma_shape = 1

            group_size = sample_gamma_reject_out_of_range(gamma_shape, min_val, max_val, 1, True, True)

        current_total_sum = sum(group_sizes)
        if (current_total_sum + group_size) > num_inbound_tourists:
            group_sizes.append(num_inbound_tourists - current_total_sum)
        else:
            group_sizes.append(group_size)

    if visualise:
        plt.figure(8)
        plt.hist(group_sizes, bins=100)
        plt.xlabel("Group sizes")
        plt.ylabel("Frequency")
        plt.title("Distribution of Group Sizes")
        plt.show(block=False)

    return group_sizes

def sample_gamma(gamma_shape, min, max, k = 1, returnInt = False):
    gamma_scale = (max - min) / (gamma_shape * k)

    sample = np.random.gamma(gamma_shape, gamma_scale)

    if returnInt:
        return round(sample)
    
    return sample

def sample_gamma_reject_out_of_range(gamma_shape, min, max, k = 1, returnInt = False, useNp = False):
    if useNp:
        sample = min - 1

        while sample < min or sample > max:
            sample = sample_gamma(gamma_shape, min, max, k, returnInt)
    else:
        scale = (max - min) / (gamma_shape * k)

        trunc_gamma = stats.truncnorm((min - scale) / np.sqrt(gamma_shape),
                              (max - scale) / np.sqrt(gamma_shape),
                              loc=scale, scale=np.sqrt(gamma_shape))
        
        sample = trunc_gamma.rvs()

    return sample
