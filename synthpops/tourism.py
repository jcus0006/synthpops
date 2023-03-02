import numpy as np
import random

def generate_tourism(inbound_aggregates, outbound_aggregates, accom_capacities, group_size_dist, gender_dist, age_groups_dist, quarter_dist, duration_dist, accom_type_dist, purpose_dist):
    accommodations = generate_accommodation(accom_capacities)

    tourists = generate_inbound_tourists(inbound_aggregates["total_inbound_tourists"], gender_dist, age_groups_dist, quarter_dist, duration_dist, purpose_dist, accom_type_dist)

    return accommodations

def generate_accommodation(accom_capacities):
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
            gamma_shape = 0.5

            # Compute the scale parameter for the gamma distribution
            gamma_scale = (max_beds - min_beds) / gamma_shape

            # Generate bed counts using rejection sampling (reject any samples not within range)
            bed_counts = []
            while len(bed_counts) < total_units:
                # Generate a gamma-distributed sample
                sample = np.random.gamma(gamma_shape, gamma_scale)
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

        # plt.figure(1)
        # plt.hist(bed_counts, bins=30)
        # plt.xlabel("Total number of beds")
        # plt.ylabel("Frequency")
        # plt.title("Distribution of number of beds")
        # plt.show()

        # create a dictionary of hotel IDs and their corresponding bed counts
        accom_beds = {i: bed_counts[i] for i in range(total_units)}

        # print out the dictionary of hotel IDs and their corresponding bed counts
        accomodations[accom_id] = accom_beds

    return accomodations

def generate_inbound_tourists(num_inbound_tourists, gender_dist, age_groups_dist, quarter_dist, duration_dist, purpose_dist, accom_type_dist):
    tourists = {}

    age_ranges = [(age_groups_dist[0][0], age_groups_dist[0][1]), (age_groups_dist[1][0], age_groups_dist[1][1]), (age_groups_dist[2][0], age_groups_dist[2][1]), (age_groups_dist[3][0], age_groups_dist[3][1])]
    age_groups_flat_dist = [age_groups_dist[0][2], age_groups_dist[1][2], age_groups_dist[2][2], age_groups_dist[3][2]] # [(0,24), (25,44), (45,64), (65,100)]

    quarter_ranges = [(quarter_dist[0][0], quarter_dist[0][1]), (quarter_dist[1][0], quarter_dist[1][1]), (quarter_dist[2][0], quarter_dist[2][1]), (quarter_dist[3][0], quarter_dist[3][1])]
    quarter_flat_dist = [quarter_dist[0][2], quarter_dist[1][2], quarter_dist[2][2], quarter_dist[3][2]] # [q1, q2, q3, q4]

    duration_ranges = [(duration_dist[0][0], duration_dist[0][1]), (duration_dist[1][0], duration_dist[1][1]), (duration_dist[2][0], duration_dist[2][1])]
    duration_flat_dist = [duration_dist[0][2], duration_dist[1][2], duration_dist[2][2]] # [(1,3), (4,6), (7,30)]

    purpose_flat_dist = [purpose_dist[0][1], purpose_dist[1][1], purpose_dist[2][1], purpose_dist[3][1]] # [hol, business, familyvisit, other]

    accom_type_flat_dist = [accom_type_dist[0][1], accom_type_dist[1][1], accom_type_dist[2][1]] # [collective, rented, non-rented]

    for i in range(num_inbound_tourists):
        tourists[i] = None

        gender = random.choices(range(len(gender_dist)), weights=gender_dist)[0]

        age_range = random.choices(range(len(age_groups_flat_dist)), weights=age_groups_flat_dist)[0] # to apply Gamma distribution especially on oldest age range

        age = random.randint(age_ranges[age_range][0], age_ranges[age_range][1]) 

        quarter_range = random.choices(range(len(quarter_flat_dist)), weights=quarter_flat_dist)[0]

        month = random.randint(quarter_ranges[quarter_range][0], quarter_ranges[quarter_range][1])

        duration_range = random.choices(range(len(duration_flat_dist)), weights=duration_flat_dist)[0] # to apply Gamma distribution especially on longest duration range

        duration = random.randint(duration_ranges[duration_range][0], duration_ranges[duration_range][1]) 

        purpose = random.choices(range(len(purpose_flat_dist)), weights=purpose_flat_dist)[0]

        accom_type = random.choices(range(len(accom_type_flat_dist)), weights=accom_type_flat_dist)[0]

        tourists[i] = { "age": age, "gender": gender, "month": month, "duration": duration, "purpose": purpose, "accom_type": accom_type}

    return tourists