import numpy as np
import random
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict
import math
import time
from calendar import monthrange
import copy
from .config import logger as log
from . import base as spb
import sciris as sc
import json

def generate_tourism(inbound_aggregates, outbound_aggregates, accom_capacities, group_size_dist, family_or_non_family_by_purpose_dist, gender_dist, age_groups_dist, quarter_dist, duration_dist, accom_type_dist, purpose_dist, year=2021, total_inbound_tourists_override=None, visualise = False):
    total_inbound_tourists = inbound_aggregates["total_inbound_tourists"]
    actual_total_inbound_tourists = total_inbound_tourists

    # visualise = True
    # total_inbound_tourists_override = 2000

    if total_inbound_tourists_override is not None:
        total_inbound_tourists = total_inbound_tourists_override
        
        total_to_actual_ratio = total_inbound_tourists / actual_total_inbound_tourists

        for i in range(4):
            accom_capacities[i][1] = math.ceil(accom_capacities[i][1] * total_to_actual_ratio) # 5
            accom_capacities[i][2] = math.ceil(accom_capacities[i][2] * total_to_actual_ratio) # 5
            accom_capacities[i][3] = math.ceil(accom_capacities[i][3] * total_to_actual_ratio) # 5
            accom_capacities[i][4] = math.ceil(accom_capacities[i][4] * total_to_actual_ratio) # 2

    print("generating synthetic tourism population of " + str(total_inbound_tourists) + " tourists")

    tourists_groups_by_day = {} # to serve as an index when generating itinerary per day. this provides all applicable tourists for day
    
    start = time.time()
    accommodations_ids_by_type, accommodations_occupancy_by_days, available_room_sizes_by_days, accoms_types_room_sizes_min_max = generate_accommodation(accom_capacities, visualise)
    print("generate_accommodation: " + str(time.time() - start))

    last_quarter_percent = quarter_dist[3][2]
    prev_dec_total_inbound_tourists = round((total_inbound_tourists * last_quarter_percent) / 3) # percent of last quarter divided by 3 months to get December only

    temp_tourists, temp_matching_tourists_ids, temp_matching_tourists_ages = generate_inbound_tourists(prev_dec_total_inbound_tourists, gender_dist, age_groups_dist, quarter_dist, duration_dist, purpose_dist, accom_type_dist, 16, visualise, 3)

    start = time.time()
    temp_group_sizes = generate_group_sizes(group_size_dist, prev_dec_total_inbound_tourists, visualise)
    temp_tourists_groups = [None] * len(temp_group_sizes)
    print("generate_group_sizes: " + str(time.time() - start))

    start = time.time()
    temp_tourists_groups, temp_tourists, temp_tourists_groups_by_days = generate_matching_tourists_groups(temp_tourists, temp_tourists_groups, tourists_groups_by_day, temp_matching_tourists_ids, temp_matching_tourists_ages, temp_group_sizes, quarter_dist, duration_dist, family_or_non_family_by_purpose_dist, accom_capacities, accommodations_ids_by_type, accommodations_occupancy_by_days, available_room_sizes_by_days, accoms_types_room_sizes_min_max, year=year, exceed_year=True)
    # groups = generate_tourist_groups(tourists, group_sizes, family_or_non_family_by_purpose_dist, use_pandas=False)
    print("generate_tourist_groups: " + str(time.time() - start))

    start = time.time()
    temp_tourists_groups = [grp for grp in temp_tourists_groups if grp["dep"] >= 0]
    temp_tourists = {id: temp_tourists[id] for grp in temp_tourists_groups for id in grp["ids"]}
    temp_tourists_groups_by_days = {day: ids for day, ids in temp_tourists_groups_by_days.items() if day > 0}
    print("remove any tourist data that does not affect current year: " + str(time.time() - start))

    start = time.time()
    tourists, matching_tourists_ids, matching_tourists_ages = generate_inbound_tourists(total_inbound_tourists, gender_dist, age_groups_dist, quarter_dist, duration_dist, purpose_dist, accom_type_dist, 16, visualise, initial_num_tourists= prev_dec_total_inbound_tourists)
    print("generate_inbound_tourists: " + str(time.time() - start))

    start = time.time()
    group_sizes = generate_group_sizes(group_size_dist, total_inbound_tourists, visualise)
    tourists_groups = [None] * len(group_sizes)
    print("generate_group_sizes: " + str(time.time() - start))

    start = time.time()
    tourists_groups, tourists, tourists_groups_by_days = generate_matching_tourists_groups(tourists, tourists_groups, tourists_groups_by_day, matching_tourists_ids, matching_tourists_ages, group_sizes, quarter_dist, duration_dist, family_or_non_family_by_purpose_dist, accom_capacities, accommodations_ids_by_type, accommodations_occupancy_by_days, available_room_sizes_by_days, accoms_types_room_sizes_min_max, year=year)
    # groups = generate_tourist_groups(tourists, group_sizes, family_or_non_family_by_purpose_dist, use_pandas=False)
    print("generate_tourist_groups: " + str(time.time() - start))

    start = time.time()
    temp_tourists.update(tourists)
    tourists = temp_tourists

    tourists = dict(sorted(tourists.items(),key=lambda x:x[0],reverse = False))

    temp_tourists_groups.extend(tourists_groups)
    tourists_groups = temp_tourists_groups

    for day, ids in temp_tourists_groups_by_days.items():
        tourists_groups_by_days[day].extend(ids)

    print("merge data from december of previous year with data from current year: " + str(time.time() - start))

    return tourists, tourists_groups, tourists_groups_by_days, accommodations_ids_by_type
    # return accommodations

def generate_accommodation(accom_capacities, visualise = False):
    accomodations_ids_by_type, accommodations_occupancy_by_ids_by_type, available_room_sizes_by_type, accommodations_occupancy_by_days, available_room_sizes_by_days, accoms_types_room_sizes_min_max = {}, {}, {}, {}, {}, {}
    
    for index, accom in enumerate(accom_capacities): # 1: hotel, 2: hostel/guest house, 3: tourist village, 4: rental accommodation
        # accommodation type
        accom_type = accom[0]

        # total number of units
        total_units = accom[1]

        # total number of beds across all units
        total_beds = accom[2]

        # minimum and maximum number of beds
        min_beds = accom[3]
        max_beds = accom[4]

        # minimum and maximum room size (num of beds per room)
        min_room_size = accom[5]
        max_room_size = accom[6]

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
                while sum(bed_counts) != total_beds:
                    bed_counts_len = len(bed_counts)

                    for i in range(bed_counts_len):
                        temp_new_bed_count = bed_counts[i] + to_add
                        
                        if min_beds <= temp_new_bed_count <= max_beds:
                            bed_counts[i] = temp_new_bed_count
                        else:
                            if to_add > 0:
                                if sum(bed_counts) + max_beds > total_beds:
                                    new_bed_count = total_beds - (sum(bed_counts) + max_beds)
                                else:
                                    new_bed_count = max_beds

                                bed_counts.append(new_bed_count)

                        if sum(bed_counts) == total_beds:
                            break              
                            
        if visualise:
            plt.figure(index+1)
            plt.hist(bed_counts, bins=30)
            plt.xlabel("Total number of beds")
            plt.ylabel("Frequency")
            plt.title("Distribution of number of beds")

            plt.show(block=False)

        # create a dictionary of accom IDs and their corresponding bed counts
        accom_beds = {i: bed_counts[i] for i in range(total_units)}

        accom_rooms_by_id, occupancy_rooms_by_id, available_room_sizes = {}, {}, {}

        accom_type_min = 0
        accom_type_max = 0
        
        for accom_id, bed_count in accom_beds.items():
            accom_rooms = {}
            accom_rooms_occupancy = {}
            room_id = 0
            min_sample_size = 0
            max_sample_size = 0
            while bed_count > 0:
                sampled_room_size = sample_gamma_reject_out_of_range(0.5, min_room_size, max_room_size, 1, True, True)

                if bed_count - sampled_room_size < 0:
                    sampled_room_size = bed_count

                if min_sample_size == 0 and max_sample_size == 0:
                    min_sample_size = sampled_room_size
                    max_sample_size = sampled_room_size
                else:
                    if sampled_room_size > max_sample_size:
                        max_sample_size = sampled_room_size

                    if sampled_room_size < min_sample_size:
                        min_sample_size = sampled_room_size

                if accom_type_min == 0 and accom_type_max == 0:
                    accom_type_min = min_sample_size
                    accom_type_max = max_sample_size
                else:
                    if max_sample_size > accom_type_max:
                        accom_type_max = max_sample_size

                    if min_sample_size < accom_type_min:
                        accom_type_min = min_sample_size         

                accom_rooms[room_id] = sampled_room_size
                accom_rooms_occupancy[room_id] = (sampled_room_size, -1, -1) # room_size, group_id, sub_group_id

                bed_count -= sampled_room_size

                room_id += 1

            for i in range(1, max_sample_size + 1): # mark room size as available, if exact or larger match is found (assignment will prefer exact room sizes, but if 1 is not found, larger room sizes are considered)
                if i not in available_room_sizes:
                    available_room_sizes[i] = []
            
                if i >= min_sample_size:
                    if accom_id not in available_room_sizes[i]:
                        available_room_sizes[i].append(accom_id)

            accom_rooms_by_id[accom_id] = OrderedDict(sorted(accom_rooms.items(), key=lambda kv: kv[1]))

            occupancy_rooms_by_id[accom_id] = {k: v for k, v in sorted(accom_rooms_occupancy.items(), key=lambda item: item[1][0])} # sort by room_size (required for sequential access later)

        accoms_types_room_sizes_min_max[accom_type] = (accom_type_min, accom_type_max)
        accomodations_ids_by_type[accom_type] = accom_rooms_by_id
        accommodations_occupancy_by_ids_by_type[accom_type] = occupancy_rooms_by_id
        available_room_sizes_by_type[accom_type] = available_room_sizes

    for i in range(-31, 365): # zero-based // starts from 1 month prior
        day = i + 1
        start = time.time()
        accommodations_occupancy_by_days[day] = copy.deepcopy(accommodations_occupancy_by_ids_by_type)
        available_room_sizes_by_days[day] = copy.deepcopy(available_room_sizes_by_type)
        # print("available room sizes by day (deep copy): Day " + str(day) + " - " + str(time.time() - start))

    return accomodations_ids_by_type, accommodations_occupancy_by_days, available_room_sizes_by_days, accoms_types_room_sizes_min_max

def generate_inbound_tourists(num_inbound_tourists, gender_dist, age_groups_dist, quarter_dist, duration_dist, purpose_dist, accom_type_dist, min_ref_person_age = 16, visualise = False, quarter_override=None, initial_num_tourists = 0):
    tourists, matching_tourists_ids, matching_tourists_ages = {}, {}, {}

    gender_options = [index for index, g in enumerate(gender_dist)]

    age_ranges_options = [index for index, a in enumerate(age_groups_dist)]
    age_ranges_gamma = [(age_groups_dist[0][0], age_groups_dist[0][1], age_groups_dist[0][3]), (age_groups_dist[1][0], age_groups_dist[1][1], age_groups_dist[1][3]), (age_groups_dist[2][0], age_groups_dist[2][1], age_groups_dist[2][3]), (age_groups_dist[3][0], age_groups_dist[3][1], age_groups_dist[3][3])]
    age_groups_flat_dist = [age_groups_dist[0][2], age_groups_dist[1][2], age_groups_dist[2][2], age_groups_dist[3][2]] # [(0,24), (25,44), (45,64), (65,100)]

    quarter_ranges_options = [index for index, q in enumerate(quarter_dist)]
    quarter_flat_dist = [quarter_dist[0][2], quarter_dist[1][2], quarter_dist[2][2], quarter_dist[3][2]] # [q1, q2, q3, q4]

    duration_ranges_options = [index for index, d in enumerate(duration_dist)]
    duration_flat_dist = [duration_dist[0][2], duration_dist[1][2], duration_dist[2][2]] # [(1,3), (4,6), (7,30)]

    purpose_options = [index+1 for index, p in enumerate(purpose_dist)]
    purpose_flat_dist = [purpose_dist[0][1], purpose_dist[1][1], purpose_dist[2][1], purpose_dist[3][1]] # [hol, business, familyvisit, other]

    accom_type_options = [index+1 for index, ac in enumerate(accom_type_dist)]
    accom_type_flat_dist = [accom_type_dist[0][1], accom_type_dist[1][1], accom_type_dist[2][1], accom_type_dist[3][1]] # [collective, rented, non-rented] scrapped, now using [hotel, guesthouse, touristvillage, self-catering]

    for i in range(num_inbound_tourists):
        tourist_id = i + initial_num_tourists # to handle previous December

        tourists[tourist_id] = None

        gender = np.random.choice(gender_options, size=1, p=gender_dist)[0]

        age_range = np.random.choice(age_ranges_options, size=1, p=age_groups_flat_dist)[0]

        # age_gamma_shape = 1
        age_gamma_shape = age_ranges_gamma[age_range][2]

        if age_gamma_shape == 0: # use uniform distribution if 0, or gamma distribution if not
            age = random.randint(age_ranges_gamma[age_range][0], age_ranges_gamma[age_range][1])
        else:
            age = sample_gamma_reject_out_of_range(age_gamma_shape, age_ranges_gamma[age_range][0], age_ranges_gamma[age_range][1], 1, True, True)

        if quarter_override is None:
            quarter_range = np.random.choice(quarter_ranges_options, size=1, p=quarter_flat_dist)[0]
        else:
            quarter_range = quarter_override

        # month = random.randint(quarter_ranges[quarter_range][0], quarter_ranges[quarter_range][1])

        duration_range = np.random.choice(duration_ranges_options, size=1, p=duration_flat_dist)[0]

        # duration_gamma_shape = duration_ranges_gamma[duration_range][2]

        # if duration_gamma_shape == 0: # use uniform distribution if 0, or gamma distribution if not
        #     duration = random.randint(duration_ranges_gamma[duration_range][0], duration_ranges_gamma[duration_range][1]) 
        # else:
        #     duration = sample_gamma_reject_out_of_range(duration_gamma_shape, duration_ranges_gamma[duration_range][0], duration_ranges_gamma[duration_range][1], 1, True, True)

        purpose = np.random.choice(purpose_options, size=1, p=purpose_flat_dist)[0]

        accom_type = np.random.choice(accom_type_options, size=1, p=accom_type_flat_dist)[0]

        tourists[tourist_id] = { "age": age, "gender": gender, "quarter": quarter_range, "duration": duration_range, "purpose": purpose, "accom_type": accom_type, "group_id": -1, "sub_group_id": -1}
        # tourists_uids.append(i)
        # tourists_ages_by_uid.append(age)

        matching_key = (quarter_range, duration_range, purpose, accom_type)

        if matching_key not in matching_tourists_ids:
            matching_tourists_ids[matching_key] = []
            matching_tourists_ages[matching_key] = []
        
        matching_tourists_ids[matching_key].append(tourist_id)
        matching_tourists_ages[matching_key].append(age)

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

    return tourists, matching_tourists_ids, matching_tourists_ages

def generate_group_sizes(group_size_dist, num_inbound_tourists, visualise=False):
    ranges = [(group_size_dist[0][0], group_size_dist[0][1]), (group_size_dist[1][0], group_size_dist[1][1]), (group_size_dist[2][0], group_size_dist[2][1]), (group_size_dist[3][0], group_size_dist[3][1])]
    percentages = [group_size_dist[0][2], group_size_dist[1][2], group_size_dist[2][2], group_size_dist[3][2]] # [(1,1), (2,2), (3,5), (6,10)]

    group_sizes = []
    for range_, percent in zip(ranges, percentages):
        if range_[0] == range_[1]:
            # if the range is just a single value, add it to the group sizes
            group_sizes += [range_[0]] * round((percent * num_inbound_tourists) / range_[0])
        else:
            # if the range has multiple values, weight the lower end more
            min_range = range_[0]
            max_range = range_[1]

            size_range = range(min_range, max_range + 1) # size range e.g. range(3, 6)
            total_percent = sum(1/i for i in size_range) # sum of the inverse of each group size. e.g. 1/3 + 1/4 + 1/5 = 0.78
            weights =  [1 / (size * total_percent) for size in size_range] # normalize e.g. 1 / (3 * 0.78), 1 / (4 * 0.78). 1 / (5 * 0.78)
            
            # for each "size" in size range (e.g. 3, 4, 5), create "weight * percent * num_inbound_toursts" indices and assign "size" into each index
            sizes = [size for size in size_range for _ in range(round((weights[size-range_[0]] * (percent * num_inbound_tourists)) / size))]
            group_sizes += sizes # add to original group_sizes array

    # when using "round", there might still be someone left to assign. add one to as many random groups as the remaining
    if sum(group_sizes) < num_inbound_tourists:
        remainder = num_inbound_tourists - sum(group_sizes)

        group_sizes_indices = [index for index, _ in enumerate(group_sizes)]

        random_group_size_indices = np.random.choice(group_sizes_indices, size=remainder, replace=False)

        for index in random_group_size_indices:
            group_sizes[index] += 1

    # when using "round", there might still be some extra sizes, in which case will decrease 1 from as many groups as the extra assigned 
    if sum(group_sizes) > num_inbound_tourists:
        extra = sum(group_sizes) - num_inbound_tourists

        group_sizes_indices = [index for index, size in enumerate(group_sizes) if size > 1]

        random_group_size_indices = np.random.choice(group_sizes_indices, size=extra, replace=False)

        for index in random_group_size_indices:
            group_sizes[index] -= 1

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

def generate_matching_tourists_groups(tourists, groups, tourists_groups_by_day, matching_tourists_by_ids, matching_tourists_by_ages, group_sizes, quarter_dist, duration_dist, family_or_non_family_by_purpose_dist, accom_capacities, accommodations_ids_by_type, accommodations_occupancy_by_days, available_room_sizes_by_days, accoms_types_room_sizes_min_max, fam_exp_rate=0.1, non_fam_relative_exp_rate=0.1, min_ref_person_age=16, year=2021, exceed_year=False):       
    fam_age_size_range = np.arange(101) # size range e.g. range(0, 100 + 1)
    fam_age_weights = np.exp(-fam_exp_rate * fam_age_size_range)
    fam_age_weights /= np.sum(fam_age_weights) # normalize the weights so they sum to 1

    # duration_ranges_gamma = [(duration_dist[i][0], duration_dist[i][1], duration_dist[i][3]) for i in range(len(duration_dist) + 1)]

    days_by_month = {}
    for i in range(12):
        month = i+1
        _, days_in_month = monthrange(year, month)
        days_by_month[month] = days_in_month

    min_day = -31
    max_day = 365 + 1    
    # if not exceed_year:
    #     min_day = 1
    #     max_day = 365 + 1
    # else:
    #     min_day = -31
    #     max_day = 30

    for i in range(min_day, max_day):
        tourists_groups_by_day[i] = []

    group_sizes = sorted(group_sizes, reverse=True)

    num_of_tourists = len(tourists)
    num_of_groups = len(group_sizes)
    matching_tourists_weights = {matching_key:len(matching_tourists)/num_of_tourists for matching_key, matching_tourists in matching_tourists_by_ids.items()}

    reference_persons = {} # key: id, value: matching_key
    matching_key_index = 0
    for matching_key, matching_tourists_ids in matching_tourists_by_ids.items():
        matching_tourists_ids = np.array(matching_tourists_ids) # convert to numpy array for faster numerical computations
        matching_tourists_ages = np.array(matching_tourists_by_ages[matching_key])
        matching_tourists_weight = matching_tourists_weights[matching_key]
        num_of_ref_persons = round(num_of_groups * matching_tourists_weight)

        if num_of_ref_persons == 0:
            num_of_ref_persons = 1
        
        if matching_key_index+1 == len(matching_tourists_by_ids): # if reached end
            if len(reference_persons) + num_of_ref_persons != num_of_groups: # if when adding ref persons for this matching group, we don't reach or exceed the group_size length
                num_of_ref_persons = abs(num_of_groups - (len(reference_persons) + num_of_ref_persons))

        matching_tourists_adults = {index:id for index, id in enumerate(matching_tourists_ids) if tourists[id]["age"] >= min_ref_person_age}

        if len(matching_tourists_adults) > 0:
            matching_tourists_adults_indices = np.array(list(matching_tourists_adults.keys()))
            
            if len(matching_tourists_adults_indices) > num_of_ref_persons:
                sampled_reference_tourist_indices = np.random.choice(matching_tourists_adults_indices, size=num_of_ref_persons, replace=False)
            else:
                sampled_reference_tourist_indices = matching_tourists_adults_indices

            for sample_index in sampled_reference_tourist_indices:
                id = matching_tourists_adults[sample_index]
                reference_persons[id] = matching_key
                
            matching_tourists_ids = np.delete(matching_tourists_ids, sampled_reference_tourist_indices)
            matching_tourists_ages = np.delete(matching_tourists_ages, sampled_reference_tourist_indices)
            
            matching_tourists_by_ids[matching_key] = matching_tourists_ids
            matching_tourists_by_ages[matching_key] = matching_tourists_ages
        else:
            print("no adult matching tourists")

        matching_key_index += 1

    if len(reference_persons) != num_of_groups:
        if len(reference_persons) > num_of_groups: # remove the extra reference_persons and re-add to pool
            ref_persons_ids = list(reference_persons.keys())

            extra = len(reference_persons) - num_of_groups

            sample_ids_to_remove = np.random.choice(ref_persons_ids, size=extra, replace=False)

            for sample_id in sample_ids_to_remove:
                matching_key = reference_persons[sample_id]
                matching_tourists_ids = matching_tourists_by_ids[matching_key]
                matching_tourists_ages = matching_tourists_by_ages[matching_key]

                matching_tourists_ids = np.append(matching_tourists_ids, sample_id)
                matching_tourists_ages = np.append(matching_tourists_ages, tourists[sample_id]["age"])

                matching_tourists_by_ids[matching_key] = matching_tourists_ids
                matching_tourists_by_ages[matching_key] = matching_tourists_ages        

                del reference_persons[sample_id]
        else: # add the missing reference_persons and remove from pool 
            remaining = num_of_groups - len(reference_persons)

            # sort by the weights, favour the larger cohorts of tourists
            matching_tourists_weights_sorted_reversed = OrderedDict(sorted(matching_tourists_weights.items(), key=lambda kv: kv[1], reverse=True))

            # until remaining = 0, traverse matching keys by largest weight first and sample a single ref tourist from adult cohort
            while remaining > 0:
                for matching_key in matching_tourists_weights_sorted_reversed.keys():
                    matching_tourists_ids = matching_tourists_by_ids[matching_key]
                    matching_tourists_ages = matching_tourists_by_ages[matching_key]
                    matching_tourists_adults = {index:id for index, id in enumerate(matching_tourists_ids) if tourists[id]["age"] >= min_ref_person_age}
                    
                    if len(matching_tourists_adults) > 0:
                        matching_tourists_adults_indices = np.array(list(matching_tourists_adults.keys()))

                        sample_index = np.random.choice(matching_tourists_adults_indices, size=1)[0]

                        id = matching_tourists_adults[sample_index]
                        reference_persons[id] = matching_key
                            
                        matching_tourists_ids = np.delete(matching_tourists_ids, sample_index)
                        matching_tourists_ages = np.delete(matching_tourists_ages, sample_index)
                        matching_tourists_by_ids[matching_key] = matching_tourists_ids
                        matching_tourists_by_ages[matching_key] = matching_tourists_ages
                    
                        remaining -= 1

                    if remaining == 0:
                        break

    ref_ids = np.array(list(reference_persons.keys()))
    np.random.shuffle(ref_ids)

    group_sizes = sorted(group_sizes, reverse=True)

    time_sum = 0
    iter_count = 0
    groups_with_missing_tourists = {}

    for group_index, group_size in enumerate(group_sizes):
        iter_start = time.time()
        
        #start = time.time()
        group = groups[group_index]
        reference_tourist_id = ref_ids[group_index]
        group = [reference_tourist_id]
        #print("returning group in iteration: " + str(time.time() - start))
        ref_tourist = tourists[reference_tourist_id]

        group_complete = True
        if group_size > 1:
            #start = time.time()     
            # ref_tourist = remaining_tourists_df.loc[reference_tourist_id]
            #print("returing ref_tourist by reference_tourist_id: " + str(time.time() - start))

            matching_key = (ref_tourist["quarter"], ref_tourist["duration"], ref_tourist["purpose"], ref_tourist["accom_type"])

            matching_tourists_ids = matching_tourists_by_ids[matching_key]
            matching_tourists_ages = matching_tourists_by_ages[matching_key]

            matching_tourists_ages = np.array(matching_tourists_ages)

            if len(matching_tourists_ids) == (group_size-1): # exactly the same number of matching tourists, add them as is
                group.extend(matching_tourists_ids.copy())

                matching_tourists_ids = []
                matching_tourists_ages = []      
            elif len(matching_tourists_ids) < (group_size-1): # not enough, these will have to be marked to be filled in at the end
                group.extend(matching_tourists_ids.copy())
                
                groups_with_missing_tourists[group_index] = (group_size-1) - len(matching_tourists_ids)

                matching_tourists_ids = []
                matching_tourists_ages = []

                group_complete = False
            else: # found enough matching tourists, must take group_size - 1
                #start = time.time()
                fam_or_non_fam_by_purpose = [dist for dist in family_or_non_family_by_purpose_dist if dist[0] == ref_tourist["purpose"]][0]
                fam_or_non_fam_by_purpose_percent = [fam_or_non_fam_by_purpose[1], fam_or_non_fam_by_purpose[2]] # [family, nonfamily]
                fam_or_non_fam_by_purpose_options = np.array([0, 1]) # [family, nonfamily]

                fam_or_non_fam = np.random.choice(fam_or_non_fam_by_purpose_options, size=1, p=fam_or_non_fam_by_purpose_percent)[0]
                #print("picking fam_or_non_fam: " + str(time.time() - start))

                #start = time.time()
                matching_tourists_ages_indices = np.array([index for index, _ in enumerate(matching_tourists_ages)])
                #print("generating matching_tourists_ages_indices: " + str(time.time() - start))

                #start = time.time()
                if fam_or_non_fam == 0: # family tourism
                    weights = [fam_age_weights[age] for age in matching_tourists_ages] # get the weight percentage for each age
                    weights = weights/np.sum(weights) # normalize the weights so they sum to 1           
                    partial_sampled_ages_indices = np.random.choice(matching_tourists_ages_indices, size=(group_size-1), replace=False, p=weights)
                else: # non family tourism
                    weights = np.exp(-non_fam_relative_exp_rate * np.abs(matching_tourists_ages - ref_tourist["age"])) # favour similar ages - calculate weights based on the distance from the target age
                    # weights = np.exp(-0.5 * (matching_tourists_ages - ref_tourist["age"])**2) # favour similar ages - calculate weights based on the distance from the target age
                    weights /= np.sum(weights) # normalize the weights so they sum to 1
                    partial_sampled_ages_indices = np.random.choice(matching_tourists_ages_indices, size=(group_size-1), replace=False, p=weights)

                #print("generating matching_tourists_ages_indices: " + str(time.time() - start))

                partial_sampled_ids = [matching_tourists_ids[i] for i in partial_sampled_ages_indices]

                group.extend(partial_sampled_ids)

                matching_tourists_ids = np.delete(matching_tourists_ids, partial_sampled_ages_indices)
                matching_tourists_ages = np.delete(matching_tourists_ages, partial_sampled_ages_indices)

            matching_tourists_by_ids[matching_key] = matching_tourists_ids
            matching_tourists_by_ages[matching_key] = matching_tourists_ages

        # sample departure day and arrival day based on "quarter" and "duration"

        if not exceed_year:
            qt_range = quarter_dist[ref_tourist["quarter"]]
            qt_min, qt_max = qt_range[0], qt_range[1]

            month = np.random.choice(np.arange(qt_min, qt_max+1, 1), size=1)[0]
        else:
            month = 12

        num_days_in_month = days_by_month[month]

        arrival_day_in_month = np.random.choice(np.arange(1, num_days_in_month + 1, 1), size=1)[0]

        arrival_day = 0
        for m, days_in_month in days_by_month.items():
            if m < month:
                arrival_day += days_in_month

        arrival_day += arrival_day_in_month

        if exceed_year:
            arrival_day -= 365 # 1 based

        dur_range = duration_dist[ref_tourist["duration"]]
        dur_min, dur_max = dur_range[0], dur_range[1]

        duration_gamma_shape = dur_range[3]

        if duration_gamma_shape == 0: # use uniform distribution if 0, or gamma distribution if not 0 (to favour smaller durations)
            duration = random.randint(dur_min, dur_max) 
        else:
            duration = sample_gamma_reject_out_of_range(duration_gamma_shape, dur_min, dur_max, 1, True, True)

        departure_day = arrival_day + duration

        group_rooms_tourist_ids = []
        group_accom = {}

        if group_complete:
            group_rooms_tourist_ids, group_accom = assign_group_into_accommodation_and_room(group, group_size, group_index, accom_capacities, ref_tourist, arrival_day, departure_day, accommodations_ids_by_type, available_room_sizes_by_days, accommodations_occupancy_by_days, accoms_types_room_sizes_min_max, exceed_year)

        #start = time.time()
        groups[group_index] = {"ids": group, "ref_tour_id": reference_tourist_id, "arr": arrival_day, "dep": departure_day, "purpose": ref_tourist["purpose"], "accom_type": ref_tourist["accom_type"], "sub_groups_ids": group_rooms_tourist_ids, "accom": group_accom}
        #print("adding group to groups: " + str(time.time() - start))

        if group_complete:
            for roomid, tourist_ids in enumerate(group_rooms_tourist_ids):
                for tourist_id in tourist_ids:
                    this_tourist = tourists[tourist_id]
                    this_tourist["group_id"] = group_index
                    this_tourist["sub_group_id"] = roomid

            for day in range(arrival_day, departure_day+1):
                if exceed_year or day <= 365:
                    tourists_groups_by_day[day].append(group_index)
                else:
                    break

        iter_count += 1
        duration = time.time() - iter_start
        time_sum += duration
        avg_time = time_sum / iter_count

        print("full iteration of iter count: " + str(iter_count) + ", group size: " + str(group_size) + ": time taken: " + str(duration) + ", average time: " + str(avg_time))

    if sum(list(groups_with_missing_tourists.values())) > 0: # handle groups that still need some tourists to be assigned to them to be full, in this case purely at random
        matching_groups_non_empty = [matching_group for matching_group in matching_tourists_by_ids.values() if len(matching_group) > 0]
        remaining_tourists_ids = np.array([id for matching_group in matching_groups_non_empty for id in matching_group])

        np.random.shuffle(remaining_tourists_ids)

        for groupindex, num_missing_tourists in groups_with_missing_tourists.items():
            group_dict = groups[groupindex]
            group = group_dict["ids"]

            sampled_ids = remaining_tourists_ids[:num_missing_tourists]
            remaining_tourists_ids = remaining_tourists_ids[num_missing_tourists:]

            group.extend(sampled_ids)

            group_rooms_tourist_ids = []
            group_accom = {}

            group_rooms_tourist_ids, group_accom = assign_group_into_accommodation_and_room(group, len(group), groupindex, accom_capacities, ref_tourist,  group_dict["arr"], group_dict["dep"], accommodations_ids_by_type, available_room_sizes_by_days, accommodations_occupancy_by_days, accoms_types_room_sizes_min_max, exceed_year)

            group_dict["sub_groups_ids"] = group_rooms_tourist_ids
            group_dict["accom"] = group_accom

            for roomid, tourist_ids in enumerate(group_rooms_tourist_ids):
                for tourist_id in tourist_ids:
                    this_tourist = tourists[tourist_id]
                    this_tourist["group_id"] = group_index
                    this_tourist["sub_group_id"] = roomid

            for day in range(arrival_day, departure_day+1):
                if exceed_year or day <= 365:
                    if group_index not in tourists_groups_by_day[day]:
                        tourists_groups_by_day[day].append(group_index)
                else:
                    break

    return groups, tourists, tourists_groups_by_day

def assign_group_into_accommodation_and_room(group, group_size, group_index, accom_capacities, ref_tourist, arrival_day, departure_day, accommodations_ids_by_type, available_room_sizes_by_days, accommodations_occupancy_by_days, accoms_types_room_sizes_min_max, exceed_year):
    group_clone = np.array(copy.deepcopy(group))

    # split into groups here
    accom_capacity = accom_capacities[ref_tourist["accom_type"] - 1] # index 0 based
    # min_room_size = accom_capacity[5]
    # max_room_size = accom_capacity[6]
    min_room_size, max_room_size = accoms_types_room_sizes_min_max[ref_tourist["accom_type"]]

    gamma_shape = accom_capacity[7]
    # gamma_shape = 1

    accom_type = ref_tourist["accom_type"]

    group_room_sizes =  random_group_partition(group_size, min_room_size, max_room_size, gamma_shape)

    np.random.shuffle(group_clone)
    
    group_accom = {}
    group_rooms_tourist_ids = []

    for group_room_size in group_room_sizes:
        group_indices = [index for index, _ in enumerate(group_clone)]
        sampled_indices = np.random.choice(group_indices, size=group_room_size, replace=False)
        sampled_ids = [group_clone[i] for i in sampled_indices]
        group_rooms_tourist_ids.append(sampled_ids)

        group_clone = np.delete(group_clone, sampled_indices)

    picked_accoms = []
    accom_not_allocated = {}
    room_not_allocated = {}
    for sub_group_index, sub_group_room in enumerate(group_rooms_tourist_ids):
        sub_group_size = len(sub_group_room)
        available_accoms = set()
        picked_accom_id = -1
        picked_room_id = -1
        picked_room_size = 0
        
        if len(picked_accoms) > 0:
            # check if already picked accoms are also available for the second/third sub_group, etc
            # i.e. give preference to already picked accommodation. room not assigned here, but later (avoid double code / have 1 assignment process)
            # using "accommodations_occupancy_by_days" only because going through the rooms solely in the context of picked_accoms is still expected to be relatively fast
            for acc_id in picked_accoms:
                # accom rooms in acc_id for arrival_day, then need to check for all other days
                # this section will not update the global collections but simply pick a room, if available
                accom_rms = accommodations_occupancy_by_days[arrival_day][accom_type][acc_id] 

                # traverse all rooms in accommodation. room_details: (room_size, group_id, sub_group_id) to pick a room
                for room_id, room_details in accom_rms.items():
                    room_size = room_details[0]
                    grp_id = room_details[1]

                    # if room size matches sub group length (by order of room size, so smaller rooms that are equal to sub_group_size will be picked first)
                    if room_size >= sub_group_size and grp_id == -1:            
                        picked_accom_id = acc_id
                        picked_room_id = room_id
                        picked_room_size = room_size

                        # traverse range from arrival day to departure day (bound by 31st Dec)
                        for day in range(arrival_day, departure_day + 1):
                            if exceed_year or day <= 365: # as of now, only consider 1 full year
                                picked_rm = accommodations_occupancy_by_days[day][accom_type][picked_accom_id][picked_room_id]
                                if picked_rm[1] >= 0:
                                    picked_accom_id = -1
                                    picked_room_id = -1
                                    picked_room_size = 0
                                    # print("reverted from picked_accoms")
                                    break
                            else:
                                break
                        
                        if picked_room_id >= 0: # if reset means room is not available for rest of trip, if not reset, room is available and can exit immediately
                            # update group_accom with key: (group_id, sub_group_id) and value: (accom_id, room_id)
                            group_accom[(group_index, sub_group_index, sub_group_size)] = (picked_accom_id, picked_room_id, picked_room_size)
                            picked_accoms.append(picked_accom_id)
                            # print("picked from picked_accoms. accom: " + str(picked_accom_id) + ", room: " + str(picked_room_id))
                            break                    
                
                if picked_accom_id >= 0: # if not reset, room is available and can exit immediately, if reset, keep checking
                    break
        
        if picked_accom_id == -1: # if accom/room not found for this sub group from earlier picked accommodations
            iter_count = 0
            # get list of accommodations that have rooms available for this group size on all trip days
            # using "available_room_sizes_by_days" to enable fast retrieval of accoms with available room sizes
            # the presence of accom_id for a particular room size, indicates that there is a room available, either for that size or larger
            for day in range(arrival_day, departure_day + 1):
                if exceed_year or day <= 365: # as of now, only consider 1 full year
                    accom_ids = available_room_sizes_by_days[day][accom_type][sub_group_size]
                    if iter_count == 0:
                        available_accoms = set(accom_ids) # get unique ids (all should be unique anyway)
                    else:
                        available_accoms = available_accoms.intersection(set(accom_ids)) # get matching ids (meaning available on subsequent days)
                else:
                    break

                iter_count += 1

            # if at least 1 accommodation has been found
            if len(available_accoms) > 0:
                available_accoms = np.array(list(available_accoms)) # convert to numpy array for faster computation

                np.random.shuffle(available_accoms) # shuffle

                pick_index = 0
                while picked_room_id == -1 and pick_index <= len(available_accoms) - 1:
                    accom_id = available_accoms[pick_index] # pick next one (still random as shuffled)

                    # accom rooms in accom_id for arrival_day, then need to check for all other days
                    # this section will not update the global collections but simply pick a room, if available
                    accom_rooms = accommodations_occupancy_by_days[arrival_day][accom_type][accom_id] 

                    # traverse all rooms in accommodation. room_details: (room_size, group_id, sub_group_id) to pick a room
                    for room_id, room_details in accom_rooms.items():
                        room_size = room_details[0]
                        grp_id = room_details[1]

                        # if room size is smaller or equal to sub group length (by order of room size, so smaller rooms that are equal to sub_group_size will be picked first)
                        if room_size >= sub_group_size and grp_id == -1:
                            picked_accom_id = accom_id
                            picked_room_id = room_id
                            picked_room_size = room_size

                            # traverse range from arrival day + 1 (as arrival day was used in initial check) and departure day (bound by 31st Dec)
                            for day in range(arrival_day + 1, departure_day + 1):
                                if exceed_year or day <= 365: # as of now, only consider 1 full year
                                    picked_rm = accommodations_occupancy_by_days[day][accom_type][picked_accom_id][picked_room_id]
                                    if picked_rm[1] >= 0:
                                        picked_accom_id = -1
                                        picked_room_id = -1
                                        picked_room_size = 0
                                        # print("reverted from general allocation")
                                        break
                                else:
                                    break

                            if picked_room_id >= 0: # if reset means room is not available for rest of trip, if not reset, room is available and can exit immediately
                                # update group_accom with key: (group_id, sub_group_id) and value: (accom_id, room_id)
                                group_accom[(group_index, sub_group_index, sub_group_size)] = (picked_accom_id, room_id, picked_room_size)
                                picked_accoms.append(picked_accom_id)
                                # print("picked from general allocation. accom: " + str(picked_accom_id) + ", room: " + str(picked_room_id))
                                break

                    pick_index += 1
            else:
                accom_not_allocated[(group_index, sub_group_index)] = sub_group_room
                # no_accom_found[(group_index, sub_group_index)] = sub_group_room

        if picked_room_id >= 0:
            # traverse range from arrival day and departure day (bound by 31st Dec) and update both collections 
            for day in range(arrival_day, departure_day + 1):
                if exceed_year or day <= 365: # as of now, only consider 1 full year
                    acc_rooms = accommodations_occupancy_by_days[day][accom_type][picked_accom_id]

                    another_available = False

                    for rm_id, rm_details in acc_rooms.items():
                        rm_size = rm_details[0]
                        rm_grp_id = rm_details[1] # -1 if empty

                        # comparison gives priority to smaller sizes i.e. equal, due to ordering but then considers larger group sizes, if not matched earlier
                        # only set another_available as False, if no other room is available of the same size or larger. 
                        # ("available_room_sizes_by_days" collection is purely used for fast retrieval of accommodations for a particular room size or larger)
                        if rm_id != picked_room_id and rm_size >= sub_group_size and rm_grp_id == -1:
                            # print("accom id: " + str(picked_accom_id) + ", room id with adequate size found: " + str(rm_id))
                            another_available = True 
                            break
                    
                    # Update "accommodations_occupancy_by_days"
                    room_details = (picked_room_size, group_index, sub_group_index)

                    if acc_rooms[picked_room_id][1] >= 0:
                        print("problemos")

                    # room_details above is not updated by reference, tuples are immutable, update explicitly
                    acc_rooms[picked_room_id] = room_details

                    # accommodations_occupancy_by_days[day][accom_type][picked_accom_id] = acc_rooms

                    # Update "available_room_sizes_by_days"
                    if not another_available:
                        accom_ids = available_room_sizes_by_days[day][accom_type][sub_group_size] # get accom ids 

                        # this may be the case and is not a bug, i.e. some sizes may not have been sampled in the accommodation creation step
                        if picked_accom_id in accom_ids:
                            accom_ids.remove(picked_accom_id) # remove this accom_id. this will ensure that for next iterations, this accom will not be considered as having a room of this size
                            # print("removed: " + str(picked_accom_id) + " for day: " + str(day))
                            # available_room_sizes_by_days[day][accom_type][sub_group_size] = accom_ids
                else:
                    break
        else:
            room_not_allocated[(group_index, sub_group_index)] = sub_group_room
            # no_room_found[(group_index, sub_group_index)] = sub_group_room

    # check if any of the other allocated rooms have any space for any of the sub groups in the group
    # if after this step any tourists are still not allocated, another fallback is required. for e.g. adding new rooms to allocate everyone
    temp_room_not_allocated = copy.deepcopy(room_not_allocated)
    temp_accom_not_allocated = copy.deepcopy(accom_not_allocated)
    temp_not_allocated = temp_room_not_allocated
    temp_not_allocated.update(temp_accom_not_allocated)

    for grp_key, sub_grp in temp_not_allocated.items():
        grp_ind = grp_key[0]
        sub_grp_ind = grp_key[1]
        not_allocated_group_room = group_rooms_tourist_ids[sub_grp_ind]
        
        ids_to_allocate = len(sub_grp)
        ids_allocated = 0

        temp_group_accom = copy.deepcopy(group_accom)
        allocated_ids = []

        for matched_key, matched_accom in temp_group_accom.items():
            matched_grp_index, matched_sub_grp_index, matched_sub_grp_size = matched_key[0], matched_key[1], matched_key[2]
            matched_accom_id, matched_room_id, matched_room_size = matched_accom[0], matched_accom[1], matched_accom[2]

            if matched_sub_grp_size < matched_room_size:
                spaces_to_allocate = matched_room_size - matched_sub_grp_size
                group_room = group_rooms_tourist_ids[matched_sub_grp_index]

                for i in range(ids_to_allocate + 1):
                    if ids_allocated < ids_to_allocate and ids_allocated < spaces_to_allocate:
                        new_room_member_id = sub_grp[ids_allocated]

                        not_allocated_group_room.remove(new_room_member_id)
                        group_room.append(new_room_member_id)
                        allocated_ids.append(new_room_member_id)

                        ids_allocated += 1
                    else:
                        break

                del group_accom[matched_key]
                group_accom[matched_grp_index, matched_sub_grp_index, matched_sub_grp_size + ids_allocated] = matched_accom
        
        # this would have already been deleted by reference
        # if grp_key in room_not_allocated:
        #     sub_group = room_not_allocated[grp_key]
        # else:
        #     sub_group = accom_not_allocated[grp_key]

        # for allocated_id in allocated_ids:
        #     sub_group.remove(allocated_id)

    # if any tourists are still not allocated at this point, create new rooms in a random accommodation
    # if any other accommodations have been assigned in this context i.e. for other sub groups within group, then give those accommodations priority
    # if not, pick a random accommodation (by accom_type), and use that.
    if len(room_not_allocated) > 0 or len(accom_not_allocated):
        temp_not_allocated = room_not_allocated
        temp_not_allocated.update(accom_not_allocated)

        for grp_key, sub_grp in temp_not_allocated.items():
            # no_room_found[grp_key] = sub_grp
            if len(picked_accoms) > 0:
                # pick a random accom and add as many rooms as necessary in it
                picked_accoms = np.array(picked_accoms)
                random_accom_id = np.random.choice(picked_accoms, size=1)[0]

                arr_day_only = accommodations_occupancy_by_days[arrival_day][accom_type][random_accom_id]
                arr_day_only_temp = dict(sorted(arr_day_only.items()))

                last_room_id = list(arr_day_only_temp)[-1]

                new_room_id = last_room_id + 1
                new_room_size = len(sub_grp)

                group_accom[(grp_key[0], grp_key[1], new_room_size)] = (random_accom_id, new_room_id, new_room_size)
                arr_day_only[new_room_id] = (new_room_size, grp_key[0], grp_key[1])
                # arrival day already taken care of
                for day in range(arrival_day + 1, departure_day + 1):
                    if exceed_year or day <= 365: # as of now, only consider 1 full year
                        each_day = accommodations_occupancy_by_days[day][accom_type][random_accom_id]

                        each_day[new_room_id] = (new_room_size, grp_key[0], grp_key[1])
                    else:
                        break
                
                accom_by_type = accommodations_ids_by_type[accom_type][random_accom_id]

                accom_by_type[new_room_id] = new_room_size

                accom_by_type = OrderedDict(sorted(accom_by_type.items(), key=lambda kv: kv[1]))

                accommodations_ids_by_type[accom_type][random_accom_id] = accom_by_type
            else:
                all_accoms_by_type = accommodations_ids_by_type[accom_type]

                accom_ids_by_type = np.array(list(all_accoms_by_type.keys()))

                random_accom_id = np.random.choice(accom_ids_by_type, size=1)[0]

                arr_day_only = accommodations_occupancy_by_days[arrival_day][accom_type][random_accom_id]
                arr_day_only_temp = dict(sorted(arr_day_only.items()))

                last_room_id = list(arr_day_only_temp)[-1]

                new_room_id = last_room_id + 1
                new_room_size = len(sub_grp)

                group_accom[(grp_key[0], grp_key[1], new_room_size)] = (random_accom_id, new_room_id, new_room_size)
                arr_day_only[new_room_id] = (new_room_size, grp_key[0], grp_key[1])
                # arrival day already taken care of
                for day in range(arrival_day + 1, departure_day + 1):
                    if exceed_year or day <= 365: # as of now, only consider 1 full year
                        each_day = accommodations_occupancy_by_days[day][accom_type][random_accom_id]

                        each_day[new_room_id] = (new_room_size, grp_key[0], grp_key[1])
                    else:
                        break

                accom_by_type = accommodations_ids_by_type[accom_type][random_accom_id]

                accom_by_type[new_room_id] = new_room_size

                accom_by_type = OrderedDict(sorted(accom_by_type.items(), key=lambda kv: kv[1]))

                accommodations_ids_by_type[accom_type][random_accom_id] = accom_by_type
                
    return group_rooms_tourist_ids, group_accom

# synthpops layers methods
def populate_tourism(pop, tourists, tourists_groups, tourists_groups_by_days):
    """
    Populate all of the tourism collections. Mostly to be converted to JSON.

    Args:
        pop (sp.Pop)                   : population (synthpops)
        tourists (dict)                : key = tourist id, value = tourist info
        tourists_groups (dict)         : key = group id, value = tourist group info
        tourists_groups_by_days (dict) : key = day, value = tourists group ids applicable for each day

    """
    log.debug("Populating tourism.")

    initialize_empty_tourists(pop, len(tourists.keys()))

    initialize_empty_tourists_groups(pop, len(tourists_groups))

    initialize_empty_tourists_groups_by_days(pop, len(tourists_groups_by_days.keys()))

    touristindex = 0
    # now populate tourists
    for touristid, tourist in tourists.items():
        # kwargs = dict(tourid=touristid,
        #             groupid=tourist["group_id"],
        #             subgroupid=tourist["sub_group_id"],
        #             age=tourist["age"],
        #             gender=tourist["gender"])
        tourist = Tourist(tourid=touristid,
                    groupid=tourist["group_id"],
                    subgroupid=tourist["sub_group_id"],
                    age=tourist["age"],
                    gender=tourist["gender"])
        # tourist.set_layer_group(**kwargs)
        pop.tourists[touristindex] = sc.dcp(tourist)
        touristindex += 1
        
    # now populate tourists groups
    for groupindex, group in enumerate(tourists_groups):
        accom_info = group["accom"]
        accom_info_list = []

        for index, accom in accom_info.items():
            accom_id, room_id, room_size = accom[0], accom[1], accom[2]
            accom_info_list.append([accom_id, room_id, room_size])

        # # make sure there are enough workplaces
        # kwargs = dict(groupid=groupindex,
        #                 subgroupsmemberids=group["sub_groups_ids"],
        #                 accominfo=accom_info_list,
        #                 reftourid=group["ref_tour_id"],
        #                 arr=group["arr"],
        #                 dep=group["dep"],
        #                 purpose=group["purpose"],
        #                 accomtype=group["accom_type"])
        
        tourist_group = TouristGroup(groupid=groupindex,
                        subgroupsmemberids=group["sub_groups_ids"],
                        accominfo=accom_info_list,
                        reftourid=group["ref_tour_id"],
                        arr=group["arr"],
                        dep=group["dep"],
                        purpose=group["purpose"],
                        accomtype=group["accom_type"])
        # tourist_group.set_layer_group(**kwargs)
        pop.tourists_groups[groupindex] = sc.dcp(tourist_group)

    touristgroupsbydaysindex = 0
    # now populate tourists groups by days
    for day, groupids in tourists_groups_by_days.items():
        # kwargs = dict(dayid=day,
        #                 member_uids=groupids)
        
        tourist_group_by_day = TouristGroupsByDays(dayid=day,
                                                    member_uids=groupids)
        # tourist_group_by_day.set_layer_group(**kwargs)
        pop.tourists_groups_by_days[touristgroupsbydaysindex] = sc.dcp(tourist_group_by_day)
        touristgroupsbydaysindex += 1

    return

def initialize_empty_tourists(pop, n_tourists=None):
    """
    Array of empty tourist objects.

    Args:
        pop (sp.Pop)       : population
        n_tourists (int) : the number of tourists to initialize
    """
    if n_tourists is not None and isinstance(n_tourists, int):
        pop.n_tourists = n_tourists
    else:
        pop.n_tourists = 0

    pop.tourists = [Tourist() for t in range(pop.n_tourists)]
    return

def initialize_empty_tourists_groups(pop, n_tourists_groups=None):
    """
    Array of empty tourist groups objects.

    Args:
        pop (sp.Pop)       : population
        n_tourists_groups (int) : the number of tourists groups to initialize
    """
    if n_tourists_groups is not None and isinstance(n_tourists_groups, int):
        pop.n_tourists_groups = n_tourists_groups
    else:
        pop.n_tourists_groups = 0

    pop.tourists_groups = [TouristGroup() for t in range(pop.n_tourists_groups)]
    return

def initialize_empty_tourists_groups_by_days(pop, n_tourists_groups_by_days=None):
    """
    Array of empty tourist groups by days objects.

    Args:
        pop (sp.Pop)       : population
        n_tourists_groups_by_days (int) : the number of touristes groups by days to initialize
    """
    if n_tourists_groups_by_days is not None and isinstance(n_tourists_groups_by_days, int):
        pop.n_tourists_groups_by_days = n_tourists_groups_by_days
    else:
        pop.n_tourists_groups_by_days = 0

    pop.tourists_groups_by_days = [TouristGroupsByDays() for t in range(pop.n_tourists_groups_by_days)]
    return

class Tourist:
    def __init__(self, tourid=None, groupid=None, subgroupid=None, age=None, gender=None):
        self.tourid = tourid
        self.groupid = groupid
        self.subgroupid = subgroupid
        self.age = age
        self.gender = gender

    def to_dict(self):
        return {"tourid": self.tourid, "groupid": self.groupid, "subgroupid": self.subgroupid, "age": self.age, "gender": self.gender}
    
    @classmethod
    def to_dict_list(cls, tourist_list):
        return [t.to_dict() for t in tourist_list]
    
class TouristGroup:
    def __init__(self, groupid=None, subgroupsmemberids=None, accominfo=None, reftourid=None, arr=None, dep=None, purpose=None, accomtype=None):
        self.groupid = groupid
        self.subgroupsmemberids = subgroupsmemberids
        self.accominfo = accominfo
        self.reftourid = reftourid
        self.arr = arr
        self.dep = dep
        self.purpose = purpose
        self.accomtype = accomtype

    def to_dict(self):
        return {"groupid": self.groupid, "subgroupsmemberids": self.subgroupsmemberids, "accominfo": self.accominfo, "reftourid": self.reftourid, "arr": self.arr, "dep": self.dep, "purpose": self.purpose, "accomtype": self.accomtype}

    @classmethod
    def to_dict_list(cls, tourist_group_list):
        return [tg.to_dict() for tg in tourist_group_list]
    
class TouristGroupsByDays:
    def __init__(self, dayid=None, member_uids=None):
        self.dayid = dayid
        self.member_uids = member_uids

    def to_dict(self):
        return {"dayid": self.dayid, "member_uids": self.member_uids}

    @classmethod
    def to_dict_list(cls, tourist_group_by_days_list):
        return [tg.to_dict() for tg in tourist_group_by_days_list]

# utility methods
def sample_gamma(gamma_shape, min, max, k = 1, returnInt = False):
    if min == max:
        return min
    
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

# Generate a variable number of group_sizes in the range min_size to max_size.
# Sample from a gamma distribution. Default gamma params favour smaller values (use gamma_shape > 1 to favour larger values)
def random_group_partition(group_size, min_size, max_size, gamma_shape = 0.5, k=1):
    group_sizes = []
    
    while sum(group_sizes) < group_size:
        sampled_group_size = sample_gamma_reject_out_of_range(gamma_shape, min_size, max_size, k, True, True)

        if sum(group_sizes) + sampled_group_size > group_size:
            sampled_group_size = group_size - sum(group_sizes)

        group_sizes.append(sampled_group_size)
    
    return group_sizes