import numpy as np
import random
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
import math
import time

def generate_tourism(inbound_aggregates, outbound_aggregates, accom_capacities, group_size_dist, family_or_non_family_by_purpose_dist, gender_dist, age_groups_dist, quarter_dist, duration_dist, accom_type_dist, purpose_dist, visualise = False):
    total_inbound_tourists = inbound_aggregates["total_inbound_tourists"]
    print("generating synthetic tourism population of " + str(total_inbound_tourists) + " tourists")

    visualise = False
    # total_inbound_tourists = 1000

    start = time.time()
    accommodations = generate_accommodation(accom_capacities, visualise)
    print("generate_accommodation: " + str(time.time() - start))

    start = time.time()
    tourists, matching_tourists_ids, matching_tourists_ages = generate_inbound_tourists(total_inbound_tourists, gender_dist, age_groups_dist, quarter_dist, duration_dist, purpose_dist, accom_type_dist, 16, visualise)
    print("generate_inbound_tourists: " + str(time.time() - start))

    start = time.time()
    group_sizes = generate_group_sizes(group_size_dist, total_inbound_tourists, visualise)
    print("generate_group_sizes: " + str(time.time() - start))

    start = time.time()
    groups = generate_matching_tourists_groups(tourists, matching_tourists_ids, matching_tourists_ages, group_sizes, family_or_non_family_by_purpose_dist)
    # groups = generate_tourist_groups(tourists, group_sizes, family_or_non_family_by_purpose_dist, use_pandas=False)
    print("generate_tourist_groups: " + str(time.time() - start))

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

def generate_inbound_tourists(num_inbound_tourists, gender_dist, age_groups_dist, quarter_dist, duration_dist, purpose_dist, accom_type_dist, min_ref_person_age = 16, visualise = False):
    tourists, matching_tourists_ids, matching_tourists_ages = {}, {}, {}

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

    purpose_options = [index+1 for index, p in enumerate(purpose_dist)]
    purpose_flat_dist = [purpose_dist[0][1], purpose_dist[1][1], purpose_dist[2][1], purpose_dist[3][1]] # [hol, business, familyvisit, other]

    accom_type_options = [index+1 for index, ac in enumerate(accom_type_dist)]
    accom_type_flat_dist = [accom_type_dist[0][1], accom_type_dist[1][1], accom_type_dist[2][1]] # [collective, rented, non-rented]

    for i in range(num_inbound_tourists):
        tourists[i] = None

        gender = np.random.choice(gender_options, size=1, p=gender_dist)[0]

        age_range = np.random.choice(age_ranges_options, size=1, p=age_groups_flat_dist)[0]

        # age_gamma_shape = 1
        age_gamma_shape = age_ranges_gamma[age_range][2]

        if age_gamma_shape == 0: # use uniform distribution if 0, or gamma distribution if not
            age = random.randint(age_ranges_gamma[age_range][0], age_ranges_gamma[age_range][1])
        else:
            age = sample_gamma_reject_out_of_range(age_gamma_shape, age_ranges_gamma[age_range][0], age_ranges_gamma[age_range][1], 1, True, True)

        quarter_range = np.random.choice(quarter_ranges_options, size=1, p=quarter_flat_dist)[0]

        # month = random.randint(quarter_ranges[quarter_range][0], quarter_ranges[quarter_range][1])

        duration_range = np.random.choice(duration_ranges_options, size=1, p=duration_flat_dist)[0]

        # duration_gamma_shape = duration_ranges_gamma[duration_range][2]

        # if duration_gamma_shape == 0: # use uniform distribution if 0, or gamma distribution if not
        #     duration = random.randint(duration_ranges_gamma[duration_range][0], duration_ranges_gamma[duration_range][1]) 
        # else:
        #     duration = sample_gamma_reject_out_of_range(duration_gamma_shape, duration_ranges_gamma[duration_range][0], duration_ranges_gamma[duration_range][1], 1, True, True)

        purpose = np.random.choice(purpose_options, size=1, p=purpose_flat_dist)[0]

        accom_type = np.random.choice(accom_type_options, size=1, p=accom_type_flat_dist)[0]

        tourists[i] = { "age": age, "gender": gender, "quarter": quarter_range, "duration": duration_range, "purpose": purpose, "accom_type": accom_type}
        # tourists_uids.append(i)
        # tourists_ages_by_uid.append(age)

        matching_key = (quarter_range, duration_range, purpose, accom_type)

        if matching_key not in matching_tourists_ids:
            matching_tourists_ids[matching_key] = []
            matching_tourists_ages[matching_key] = []
        
        matching_tourists_ids[matching_key].append(i)
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

        group_sizes_indices = [index for index, _ in enumerate(group_sizes)]

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


def generate_tourist_groups(tourists, group_sizes, family_or_non_family_by_purpose_dist, fam_exp_rate=0.1, non_fam_relative_exp_rate=0.1, use_pandas=False):
    groups = [None] * len(group_sizes)

    if not use_pandas:
        remaining_tourists = tourists.copy()
    else:
        remaining_tourists_df = pd.DataFrame.from_dict(tourists, orient='index')

    fam_age_size_range = np.arange(101) # size range e.g. range(0, 100 + 1)
    fam_age_weights = np.exp(-fam_exp_rate * fam_age_size_range)
    fam_age_weights /= np.sum(fam_age_weights) # normalize the weights so they sum to 1

    remaining_tourists_non_kids_ids = np.array([id for id, tourist in tourists.items() if tourist["age"] >= 16])
    remaining_tourists_non_kids_indices = np.array([index for index, _ in enumerate(remaining_tourists_non_kids_ids)])
    groups_with_missing_tourists = {}

    # num_of_single_groups = sum([1 for group_size in group_sizes if group_size == 1])

    reference_tourists_indices = np.random.choice(remaining_tourists_non_kids_indices, size=len(group_sizes), replace=False)

    reference_tourists_ids = [remaining_tourists_non_kids_ids[index] for index in reference_tourists_indices]

    for index, id in enumerate(reference_tourists_ids):
        groups[index] = [id]

        if not use_pandas:
            del remaining_tourists[id]

    # in this case the indices will be intact, and may simply delete as is
    if use_pandas:
        remaining_tourists_df = remaining_tourists_df.drop(index=reference_tourists_ids)
    
    # remaining_tourists_non_kids_ids = np.delete(remaining_tourists_non_kids_ids, reference_tourists_indices)
    # remaining_tourists_non_kids_indices = np.delete(remaining_tourists_non_kids_indices, reference_tourists_indices)

    # non_single_group_sizes = group_sizes[num_of_single_groups+1:]

    # non_single_group_sizes_reversed = non_single_group_sizes[::-1]

    group_sizes = sorted(group_sizes, reverse=True)

    time_sum = 0
    iter_count = 0
    for group_index, group_size in enumerate(group_sizes):
        if group_size > 1:
            iter_start = time.time()

            # group_index = num_of_single_groups + (len(non_single_group_sizes) - reversed_group_index)

            # start = time.time()
            # # sample an index value, here: index does not necessarily represent position
            # temp_reference_tourist_index = np.random.choice(remaining_tourists_non_kids_indices, size=1)[0]
            # print("picking reference tourist index: " + str(time.time() - start))

            # start = time.time()
            # # find the actual index position by the index value, that will match in remaining_tourists_non_kids_ids
            # reference_tourist_index = np.where(np.in1d(remaining_tourists_non_kids_indices , [temp_reference_tourist_index]))[0][0]
            # print("finding ref tourist index: " + str(time.time() - start))

            # start = time.time()
            # reference_tourist_id = remaining_tourists_non_kids_ids[reference_tourist_index]
            # print("returing reference_tourist_id by index: " + str(time.time() - start))

            # group = [reference_tourist_id]
            
            #start = time.time()
            group = groups[group_index]
            reference_tourist_id = group[0]
            #print("returning group in iteration: " + str(time.time() - start))

            #start = time.time()
            ref_tourist = tourists[reference_tourist_id]
            # ref_tourist = remaining_tourists_df.loc[reference_tourist_id]
            #print("returing ref_tourist by reference_tourist_id: " + str(time.time() - start))

            #start = time.time()
            # # remove reference tourist id, so it does not get picked again
            # remaining_tourists.pop(reference_tourist_id, None)
            # remaining_tourists_df.drop(index=ref_tourist.name, inplace=True)

            # remaining_tourists_non_kids_ids = np.delete(remaining_tourists_non_kids_ids, reference_tourist_index)
            # remaining_tourists_non_kids_indices = np.delete(remaining_tourists_non_kids_indices, reference_tourist_index)
            # print("removing reference tourist from collections: " + str(time.time() - start))
            
            if not use_pandas:
                # start = time.time()
                matching_tourists = {id:tourist["age"] for id, tourist in remaining_tourists.items() 
                                        if tourist["quarter"] == ref_tourist["quarter"] and 
                                        tourist["duration"] == ref_tourist["duration"] and 
                                        tourist["purpose"] == ref_tourist["purpose"] and 
                                        tourist["accom_type"] == ref_tourist["accom_type"]}
                # print("matching tourists. found " + str(len(matching_tourists)) + ": " + str(time.time() - start))
            else:
                # start = time.time()
                excluded_ids = np.concatenate(groups)
                matching_mask = (~remaining_tourists_df.index.isin(excluded_ids)) & (remaining_tourists_df["quarter"] == ref_tourist["quarter"]) & (remaining_tourists_df["duration"] == ref_tourist["duration"]) & (remaining_tourists_df["purpose"] == ref_tourist["purpose"]) & (remaining_tourists_df["accom_type"] == ref_tourist["accom_type"])

                matching_tourists = remaining_tourists_df[matching_mask]
                # print("remaining_tourists_df masking: " + str(time.time() - start))

            matching_tourists_ids, matching_tourists_ages = [], []

            if len(matching_tourists) > 0:
                if not use_pandas:
                    matching_tourists_ids, matching_tourists_ages = zip(*matching_tourists.items())
                else:
                    matching_tourists_ids = matching_tourists.index.values
                    matching_tourists_ages = matching_tourists["age"].values

                matching_tourists_ids = np.array(matching_tourists_ids)
                matching_tourists_ages = np.array(matching_tourists_ages)

            if len(matching_tourists_ids) == (group_size-1): # exactly the same number of matching tourists, add them as is
                group.extend(matching_tourists_ids)

                if not use_pandas:
                    for id in matching_tourists_ids:
                        del remaining_tourists[id]
                else:
                    remaining_tourists_df.drop(index=matching_tourists.index.values, inplace=True)    

                # matching_indices = np.where(np.in1d(remaining_tourists_non_kids_ids, matching_tourists_ids))[0] # np.where returns a tuple, first element are indices that match
                # remaining_tourists_non_kids_ids = np.delete(remaining_tourists_non_kids_ids, matching_indices)
                # remaining_tourists_non_kids_indices = np.delete(remaining_tourists_non_kids_indices, matching_indices)
                
            elif len(matching_tourists_ids) < (group_size-1): # not enough, these will have to be marked to be filled in at the end
                group.extend(matching_tourists_ids)
                
                groups_with_missing_tourists[group_index] = (group_size-1) - len(matching_tourists_ids)

                if not use_pandas:
                    for id in matching_tourists_ids:
                        del remaining_tourists[id]
                else:
                    remaining_tourists_df.drop(index=matching_tourists.index.values, inplace=True)  

                # matching_indices = np.where(np.in1d(remaining_tourists_non_kids_ids, matching_tourists_ids))[0] # np.where returns a tuple, first element are indices that match
                # remaining_tourists_non_kids_ids = np.delete(remaining_tourists_non_kids_ids, matching_indices)
                # remaining_tourists_non_kids_indices = np.delete(remaining_tourists_non_kids_indices, matching_indices)
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

                if not use_pandas:
                    for id in partial_sampled_ids:
                        del remaining_tourists[id]
                else:
                    # start = time.time()     
                    df_loc_indices = remaining_tourists_df.index.get_indexer(partial_sampled_ids)
                    # print("generating df_loc_indices: " + str(time.time() - start))

                    # start = time.time()
                    labels_to_drop = remaining_tourists_df.index[df_loc_indices]
                    # print("generating labels_to_drop: " + str(time.time() - start))

                    # start = time.time()
                    remaining_tourists_df.drop(index=labels_to_drop, inplace=True)
                    # print("dropping labels_to_drop from remaining_tourists_df: " + str(time.time() - start))

                #partial_sampled_ids = np.array(partial_sampled_ids)

                #start = time.time()
                #matching_indices = np.where(np.in1d(remaining_tourists_non_kids_ids, partial_sampled_ids))[0] # np.where returns a tuple, first element are indices that match
                #print("generating matching_indices: " + str(time.time() - start))

                #start = time.time()
                #remaining_tourists_non_kids_ids = np.delete(remaining_tourists_non_kids_ids, matching_indices)
                #print("np.delete on remaining_tourists_non_kids_ids: " + str(time.time() - start))

                #start = time.time()
                #remaining_tourists_non_kids_indices = np.delete(remaining_tourists_non_kids_indices, matching_indices)
                #print("np.delete on remaining_tourists_non_kids_indices: " + str(time.time() - start))

            #start = time.time()
            groups[group_index] = group
            #print("adding group to groups: " + str(time.time() - start))
            
            iter_count += 1
            duration = time.time() - iter_start
            time_sum += duration
            avg_time = time_sum / iter_count

            print("full iteration of iter count: " + str(iter_count) + ", group size: " + str(group_size) + ": time taken: " + str(duration) + ", average time: " + str(avg_time))

    if (not use_pandas and len(remaining_tourists) > 0) or (use_pandas and len(remaining_tourists_df) > 0): # handle groups that still need some tourists to be assigned to them to be full, in this case purely at random
        if not use_pandas:
            remaining_tourists_ids = np.array([id for id in remaining_tourists.keys()])
        else:
            remaining_tourists_ids = np.array(remaining_tourists_df.index.values)

        for groupindex, num_missing_tourists in groups_with_missing_tourists.items():
            group = groups[groupindex]

            sampled_ids = np.random.choice(remaining_tourists_ids, size=num_missing_tourists, replace=False)

            group.extend(sampled_ids)

            if not use_pandas:
                for id in sampled_ids:
                    del remaining_tourists[id]
            else:
                df_loc_indices = remaining_tourists_df.index.get_indexer(sampled_ids)
                labels_to_drop = remaining_tourists_df.index[df_loc_indices]
                remaining_tourists_df.drop(index=labels_to_drop, inplace=True)  
            
            matching_indices = np.where(np.in1d(remaining_tourists_ids, sampled_ids))[0] # np.where returns a tuple, first element are indices that match
            remaining_tourists_ids = np.delete(remaining_tourists_ids, matching_indices)

            # matching_indices = np.where(np.in1d(remaining_tourists_non_kids_ids, sampled_ids))[0] # np.where returns a tuple, first element are indices that match
            # remaining_tourists_non_kids_ids = np.delete(remaining_tourists_non_kids_ids, matching_indices)
            # remaining_tourists_non_kids_indices = np.delete(remaining_tourists_non_kids_indices, matching_indices)

    return groups

def generate_matching_tourists_groups(tourists, matching_tourists_by_ids, matching_tourists_by_ages, group_sizes, family_or_non_family_by_purpose_dist, fam_exp_rate=0.1, non_fam_relative_exp_rate=0.1, min_ref_person_age=16):
    groups = [None] * len(group_sizes)

    fam_age_size_range = np.arange(101) # size range e.g. range(0, 100 + 1)
    fam_age_weights = np.exp(-fam_exp_rate * fam_age_size_range)
    fam_age_weights /= np.sum(fam_age_weights) # normalize the weights so they sum to 1

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

            sampled_reference_tourist_indices = np.random.choice(matching_tourists_adults_indices, size=num_of_ref_persons, replace=False)

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

            sample_ids_to_remove = np.random.choice(ref_persons_ids, size=extra)

            for sample_id in sample_ids_to_remove:
                matching_key = reference_persons[sample_id]
                matching_tourists_ids = matching_tourists_by_ids[matching_key]
                matching_tourists_ages = matching_tourists_by_ages[matching_key]

                matching_tourists_ids.append(sample_id)
                matching_tourists_ages.append(tourists[sample_id]["age"])

                matching_tourists_by_ids[matching_key] = matching_tourists_ids
                matching_tourists_by_ages[matching_key] = matching_tourists_ages        

                del reference_persons[sample_id]
        else: # add the missing reference_persons and remove from pool 
            remaining = num_of_groups - len(reference_persons)

            # sort by the weights, favour the larger cohorts of tourists
            matching_tourists_weights_sorted_reversed = dict(sorted(matching_tourists_weights.items(), key=matching_tourists_weights.get,reverse=True))

            # until remaining = 0, traverse matching keys by largest weight first and sample a single ref tourist from adult cohort
            while remaining > 0:
                for matching_key in matching_tourists_weights_sorted_reversed.keys():
                    matching_tourists_ids = matching_tourists_by_ids[matching_key]
                    matching_tourists_ages = matching_tourists_by_ages[matching_key]
                    matching_tourists_adults = {index:id for index, id in enumerate(matching_tourists_ids) if tourists[id]["age"] >= min_ref_person_age}
                    
                    if len(matching_tourists_adults) > 0:
                        matching_tourists_adults_indices = np.array(list(matching_tourists_adults.keys()))

                        sample_index = np.random.choice(matching_tourists_adults_indices, size=1, replace=False)

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

        if group_size > 1:
            #start = time.time()
            ref_tourist = tourists[reference_tourist_id]
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

        #start = time.time()
        groups[group_index] = group
        #print("adding group to groups: " + str(time.time() - start))
            
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
            group = groups[groupindex]

            sampled_ids = remaining_tourists_ids[:num_missing_tourists]
            remaining_tourists_ids = remaining_tourists_ids[num_missing_tourists:]

            group.extend(sampled_ids)

    return groups


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
