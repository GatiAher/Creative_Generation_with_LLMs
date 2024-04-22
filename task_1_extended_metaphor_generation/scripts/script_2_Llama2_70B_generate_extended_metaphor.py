# Script to run consolidated extended metaphor generation with Llama2 70B Chat hosted on Replicate

import tqdm
import pandas as pd
import json
import re

import pathlib
import sys

import os
# we will use replicate's hosted api
import replicate

# Add src module to path before import.
sys.path.insert(0, str(pathlib.Path('../../src')))

from file_IO_handler import get_plaintext_file_contents, save_json, load_json
from fill_string_template import get_filled_strings_from_dataframe, FilledString

###############################
# Get Access to Replicate API #
###############################

REPLICATE_API_KEY = get_plaintext_file_contents(pathlib.Path("../../REPLICATE_API_KEY.env"))
# print(REPLICATE_API_KEY)

os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_KEY

# use model hosted on Replicate server for inferencing
model = "meta/llama-2-70b-chat:2d19859030ff705a87c746f7e96eea03aefb71f166725aee39692f1476566d48"


# text completion with input prompt
def Completion(prompt, is_json=False, is_short_ans=False):
    if is_json:
        output = replicate.run(
            model,
            input={
                "system_prompt": "response in json format",
                "prompt": prompt, 
                "max_new_tokens": 1000
                }
            )
        return "".join(output)
    
    elif is_short_ans:
        output = replicate.run(
            model,
            input={
                "system_prompt": "Output one word or phrase",
                "prompt": prompt, 
                "max_new_tokens": 1000
                }
            )
        return "".join(output)
    
    output = replicate.run(
        model,
        input={
            "prompt": prompt, 
            "max_new_tokens": 1000
            }
        )
    return "".join(output)



##################
# Save Locations #
##################

# Save every call
prompt_use_relationships_output_save_to_prefix = pathlib.Path("../generations/script_run_Llama70B/prompt_use_relationships_output")
prompt_cover_subtensors_output_save_to_prefix = pathlib.Path("../generations/script_run_Llama70B/prompt_cover_subtensors_output")

# Consolidated to csv
prompt_use_relationships_extended_metaphors_output_save_to = pathlib.Path("../generations/script_run_Llama70B/prompt_use_relationships_extended_metaphors_output.csv")
prompt_cover_subtensors_extended_metaphors_output_save_to = pathlib.Path("../generations/script_run_Llama70B/prompt_cover_subtensors_extended_metaphors_output.csv")

###################
# PREPARE PROMPTS #
###################

prompt_use_relationships_style_template = get_plaintext_file_contents(pathlib.Path("../prompt_template/prompt_use_relationships_write_extended_metaphor.txt"))
prompt_cover_subtensors_style_template = get_plaintext_file_contents(pathlib.Path("../prompt_template/prompt_cover_subtensors_write_extended_metaphor.txt"))


def run_prompt_use_relationships_extended_metaphors(filled_strings):
    pre_df = {
        "level_of_difficulty": [],
        "tensor_name": [],
        "source_domain": [],
        "target_domain": [],
        "list_of_is_like": [],
        "list_of_subtensors": [], 
        "list_of_subvehicles": [],
        "list_of_extended_metaphor": [],
        "use_relationships_extended_metaphor": [],
    }

    for idx in range(len(filled_strings)):
        save_path = pathlib.Path(f"{prompt_use_relationships_output_save_to_prefix}_idx_{idx}.txt")

        level_of_difficulty = filled_strings[idx].values["level_of_difficulty"]
        tensor_name = filled_strings[idx].values["tensor_name"]
        source_domain = filled_strings[idx].values["source_domain"]
        target_domain = filled_strings[idx].values["target_domain"]
        list_of_is_like = filled_strings[idx].values["list_of_is_like"]
        list_of_subtensors = filled_strings[idx].values["list_of_subtensors"]
        list_of_subvehicles = filled_strings[idx].values["list_of_subvehicles"]
        list_of_extended_metaphor = filled_strings[idx].values["list_of_extended_metaphor"]

        pre_df["level_of_difficulty"].append(level_of_difficulty)
        pre_df["tensor_name"].append(tensor_name)
        pre_df["source_domain"].append(source_domain)
        pre_df["target_domain"].append(target_domain)
        pre_df["list_of_is_like"].append(list_of_is_like)
        pre_df["list_of_subtensors"].append(list_of_subtensors)
        pre_df["list_of_subvehicles"].append(list_of_subvehicles)
        pre_df["list_of_extended_metaphor"].append(list_of_extended_metaphor)
        
        if save_path.exists():
            print(f"Loading from {save_path}")
            with open(save_path, 'r') as file:
                out_content = file.read()
        else:
            # call model
            out_content = Completion(filled_strings[idx].filled, is_json=False)
            
            # save output
            with open(save_path, 'w+') as file:
                file.write(out_content)
            print(f"Called model and saved to {save_path}")

        pre_df["use_relationships_extended_metaphor"].append(out_content)

    df_out = pd.DataFrame(pre_df)
    df_out.to_csv(prompt_use_relationships_extended_metaphors_output_save_to)
    
    return df_out


def run_prompt_cover_subtensors_extended_metaphors(filled_strings):
    pre_df = {
        "level_of_difficulty": [],
        "tensor_name": [],
        "source_domain": [],
        "target_domain": [],
        "list_of_is_like": [],
        "list_of_subtensors": [], 
        "list_of_subvehicles": [],
        "list_of_extended_metaphor": [],
        "cover_subtensors_extended_metaphor": [],
    }

    for idx in range(len(filled_strings)):
        save_path = pathlib.Path(f"{prompt_cover_subtensors_output_save_to_prefix}_idx_{idx}.txt")

        level_of_difficulty = filled_strings[idx].values["level_of_difficulty"]
        tensor_name = filled_strings[idx].values["tensor_name"]
        source_domain = filled_strings[idx].values["source_domain"]
        target_domain = filled_strings[idx].values["target_domain"]
        list_of_is_like = filled_strings[idx].values["list_of_is_like"]
        list_of_subtensors = filled_strings[idx].values["list_of_subtensors"]
        list_of_subvehicles = filled_strings[idx].values["list_of_subvehicles"]
        list_of_extended_metaphor = filled_strings[idx].values["list_of_extended_metaphor"]

        pre_df["level_of_difficulty"].append(level_of_difficulty)
        pre_df["tensor_name"].append(tensor_name)
        pre_df["source_domain"].append(source_domain)
        pre_df["target_domain"].append(target_domain)
        pre_df["list_of_is_like"].append(list_of_is_like)
        pre_df["list_of_subtensors"].append(list_of_subtensors)
        pre_df["list_of_subvehicles"].append(list_of_subvehicles)
        pre_df["list_of_extended_metaphor"].append(list_of_extended_metaphor)
    
        if save_path.exists():
            print(f"Loading from {save_path}")
            with open(save_path, 'r') as file:
                out_content = file.read()
        else:
            # call model
            out_content = Completion(filled_strings[idx].filled, is_json=False)
            
            # save output
            with open(save_path, 'w+') as file:
                file.write(out_content)
            print(f"Called model and saved to {save_path}")

        pre_df["cover_subtensors_extended_metaphor"].append(out_content)

    df_out = pd.DataFrame(pre_df)
    df_out.to_csv(prompt_cover_subtensors_extended_metaphors_output_save_to)
    
    return df_out


def is_like(row):
    if row['subtensor_name'].endswith("s"):
        return row['subtensor_name'] + " are like " + row['subvehicle_name'] + "."
    return row['subtensor_name'] + " is like " + row['subvehicle_name'] + "."


def main():
    df = pd.read_csv(pathlib.Path("../generations/script_run_Llama70B/prompt_7_output.csv"), index_col=0)
    df['is_like'] = df.apply(is_like, axis=1)
    df['list_of_is_like'] = df[['tensor_name','target_domain', 'is_like']].groupby(['tensor_name','target_domain'])['is_like'].transform(lambda x: ' '.join(x))
    df['list_of_subtensors'] = df[['tensor_name','target_domain', 'subtensor_name']].groupby(['tensor_name','target_domain'])['subtensor_name'].transform(lambda x: ', '.join(x))
    df['list_of_subvehicles'] = df[['tensor_name','target_domain', 'subvehicle_name']].groupby(['tensor_name','target_domain'])['subvehicle_name'].transform(lambda x: ', '.join(x))
    df['list_of_extended_metaphor'] = df[['tensor_name','target_domain', 'extended_metaphor']].groupby(['tensor_name','target_domain'])['extended_metaphor'].transform(lambda x: ','.join(x))
    df_slice = df[[
        'level_of_difficulty',
        'tensor_name',
        'source_domain',
        'target_domain', 
        'list_of_is_like', 
        'list_of_subtensors', 
        'list_of_subvehicles', 
        'list_of_extended_metaphor',
    ]].drop_duplicates()

    filled_strings = get_filled_strings_from_dataframe(prompt_use_relationships_style_template, df_slice)
    df_use_relationships = run_prompt_use_relationships_extended_metaphors(filled_strings)

    filled_strings = get_filled_strings_from_dataframe(prompt_cover_subtensors_style_template, df_slice)
    df_cover_subtensors = run_prompt_cover_subtensors_extended_metaphors(filled_strings)

    print("Finished running script.")
    print(f"Final number of rows: {len(df_use_relationships)}")
    print(f"Final number of rows: {len(df_cover_subtensors)}")


if __name__ == "__main__":
    main()