# %%
########################################################################
# This files holds some helper functions to process log datas.
# Author: Farshid Asadi, farshidasadi47@yahoo.com
########## Libraries ###################################################
import re
from collections import defaultdict
import math


########## Functions ###################################################
def round_up_to_first_decimal(value):
    return math.ceil(value * 10) / 10


def parse_data_to_dict(filename):
    main_dict = {}
    with open(filename, "r") as file:
        content = file.read()
    # Split content by separator line
    sections = content.split("*" * 79)
    # Process each section
    for section in sections:
        # Extract tol_cmd and goal_bias as floats
        tol_cmd_match = re.search(r"tol_cmd:\s*([\d.]+)", section)
        goal_bias_match = re.search(r"goal_bias:\s*([\d.]+)", section)
        #
        if tol_cmd_match and goal_bias_match:
            tol_cmd = float(tol_cmd_match.group(1))
            goal_bias = float(goal_bias_match.group(1))
            # Initialize nested dictionary structure for this t
            section_dict = defaultdict(list)
            # Find all path value lines with tree node values
            for match in re.finditer(
                r"Number of tree nodes (\d+), Path value:\s*([\d.]+)", section
            ):
                tree_nodes = int(match.group(1))
                path_value = int(float(match.group(2)))
                rounded_time = round_up_to_first_decimal(tree_nodes / 1000)
                section_dict[rounded_time].append(path_value)
            # Add this section's dictionary to the main dictionary
            main_dict[(tol_cmd, goal_bias)] = dict(section_dict)
    return main_dict


def sort_dict_by_smallest_path(main_dict):
    sorted_dict = dict(
        sorted(
            main_dict.items(),
            key=lambda item: (
                min(
                    (min(paths) if paths else float("inf"))
                    for paths in item[1].values()
                )
                if item[1]
                else float("inf")
            ),  # Handle empty time_dict entries
        )
    )
    return sorted_dict


def print_entries_up_to_path_length(main_dict, max_path_length):
    # Iterate over each (tol_cmd, goal_bias) entry
    for (tol_cmd, goal_bias), time_dict in main_dict.items():
        if len(time_dict):
            if min(min(row) for row in time_dict.values()) < max_path_length:
                print(f"({tol_cmd}, {goal_bias}): {time_dict}")
            else:
                break


########## test section ################################################
if __name__ == "__main__":
    filename = "logfile.log"
    main_dict = parse_data_to_dict(filename)
    sorted_dict = sort_dict_by_smallest_path(main_dict)
    print_entries_up_to_path_length(sorted_dict, max_path_length=40)
