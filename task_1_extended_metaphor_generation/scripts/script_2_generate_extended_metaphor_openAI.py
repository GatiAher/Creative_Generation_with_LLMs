# Script to run consolidated extended metaphor generation with Open AI models

import tqdm
import pandas as pd
import json
import re

import pathlib
import sys

from openai import OpenAI
import os

# Add src module to path before import.
sys.path.insert(0, str(pathlib.Path('../../src')))

from file_IO_handler import get_plaintext_file_contents, save_json, load_json
from fill_string_template import get_filled_strings_from_dataframe, FilledString

############################
# Get Access to OpenAI API #
############################

OPENAI_API_KEY = get_plaintext_file_contents(pathlib.Path("../../OPENAI_API_KEY.env"))
# print(OPENAI_API_KEY)

MODEL_NAME = "gpt-3.5-turbo"


client = OpenAI(api_key=OPENAI_API_KEY)


def Completion(prompt, is_json=False, is_short_ans=False):
    if is_json:
        return client.chat.completions.create(
            model="gpt-3.5-turbo",
            response_format={ "type": "json_object" },
            messages=[{"role": "system", "content": "You are a helpful assistant designed to output JSON."}, {"role": "user", "content": prompt}]
        )
    elif is_short_ans:
        return client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "Output one word or phrase"}, {"role": "user", "content": prompt}]
        )
    return client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )


##################
# Save Locations #
##################

# Save every call
prompt_use_relationships_output_save_to_prefix = pathlib.Path("../generations/script_run_GPT35Turbo/prompt_use_relationships_output")
prompt_cover_subtensors_output_save_to_prefix = pathlib.Path("../generations/script_run_GPT35Turbo/prompt_cover_subtensors_output")

# Consolidated to csv
prompt_use_relationships_extended_metaphors_output_save_to = pathlib.Path("../generations/script_run_GPT35Turbo/prompt_use_relationships_extended_metaphors_output.csv")
prompt_cover_subtensors_extended_metaphors_output_save_to = pathlib.Path("../generations/script_run_GPT35Turbo/prompt_cover_subtensors_extended_metaphors_output.csv")

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
            out = Completion(filled_strings[idx].filled, is_json=False)
            out_content = out.choices[0].message.content
            
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
            out = Completion(filled_strings[idx].filled, is_json=False)
            out_content = out.choices[0].message.content
            
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
    df = pd.read_csv(pathlib.Path("../generations/script_run_GPT35Turbo/prompt_7_output.csv"), index_col=0)
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