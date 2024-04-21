# Script to run extended metaphor generation with Open AI models

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
# Target Domains #
##################

target_domains = ["Kpop", "baseball", "Iron Man", "money"]

##################
# Save Locations #
##################

# Save every call
prompt_1_output_save_to_prefix = pathlib.Path("../generations/script_run/prompt_1_output")
prompt_2_output_save_to_prefix = pathlib.Path("../generations/script_run/prompt_2_output")
prompt_3_output_save_to_prefix = pathlib.Path("../generations/script_run/prompt_3_output")
prompt_4_output_save_to_prefix = pathlib.Path("../generations/script_run/prompt_4_output")
prompt_5_output_save_to_prefix = pathlib.Path("../generations/script_run/prompt_5_output")
prompt_6_output_save_to_prefix = pathlib.Path("../generations/script_run/prompt_6_output")
prompt_7_output_save_to_prefix = pathlib.Path("../generations/script_run/prompt_7_output")


# Consolidated to csv
prompt_1_output_save_to = pathlib.Path("../generations/script_run/prompt_1_output.csv")
prompt_2_output_save_to = pathlib.Path("../generations/script_run/prompt_2_output.csv")
prompt_3_output_save_to = pathlib.Path("../generations/script_run/prompt_3_output.csv")
prompt_4_output_save_to = pathlib.Path("../generations/script_run/prompt_4_output.csv")
prompt_5_output_save_to = pathlib.Path("../generations/script_run/prompt_5_output.csv")
prompt_6_output_save_to = pathlib.Path("../generations/script_run/prompt_6_output.csv")
prompt_7_output_save_to = pathlib.Path("../generations/script_run/prompt_7_output.csv")


###################
# PREPARE PROMPTS #
###################

prompt_1_template = get_plaintext_file_contents(pathlib.Path("../prompt_template/prompt_1_get_subtensors.txt"))
prompt_2_template = get_plaintext_file_contents(pathlib.Path("../prompt_template/prompt_2_get_purpose_and_mechanism.txt"))
prompt_3_template = get_plaintext_file_contents(pathlib.Path("../prompt_template/prompt_3_annotate_text.txt"))
prompt_4_template = get_plaintext_file_contents(pathlib.Path("../prompt_template/prompt_4_create_schema.txt"))
prompt_5_template = get_plaintext_file_contents(pathlib.Path("../prompt_template/prompt_5_identify_subvehicle.txt"))
prompt_6_template = get_plaintext_file_contents(pathlib.Path("../prompt_template/prompt_6_write_analogy.txt"))
prompt_7_template = get_plaintext_file_contents(pathlib.Path("../prompt_template/prompt_7_write_poem.txt"))


def run_prompt_1(filled_strings):
    pre_df = {
        "level_of_difficulty": [],
        "tensor_name": [],
        "source_domain": [],
        "subtensor_name": [],
        "subtensor_definition": [],
    }
    
    for idx in range(len(filled_strings)):
        save_path = pathlib.Path(f"{prompt_1_output_save_to_prefix}_idx_{idx}.json")

        level_of_difficulty = filled_strings[idx].values["level_of_difficulty"]
        tensor_name = filled_strings[idx].values["tensor_name"]
        source_domain = filled_strings[idx].values["source_domain"]
        
        if save_path.exists():
            print(f"Loading from {save_path}")
            out_content = load_json(save_path)
        else:
            # call model
            out = Completion(filled_strings[idx].filled, is_json=True)
            out_content = json.loads(out.choices[idx].message.content)
            
            # save output
            save_json(out_content, save_path)
            print(f"Called model and saved to {save_path}")
        
        for obj in out_content["sub_concepts"]:
            pre_df["level_of_difficulty"].append(level_of_difficulty)
            pre_df["tensor_name"].append(tensor_name)
            pre_df["source_domain"].append(source_domain)
            # new
            pre_df["subtensor_name"].append(obj["name"])
            pre_df["subtensor_definition"].append(obj["definition"])

    df_prompt_2_input = pd.DataFrame(pre_df)
    df_prompt_2_input.to_csv(prompt_1_output_save_to)
    
    return df_prompt_2_input


def run_prompt_2(filled_strings):
    pre_df = {
        "level_of_difficulty": [],
        "tensor_name": [],
        "source_domain": [],
        "subtensor_name": [],
        "subtensor_definition": [],
        "subtensor_purpose": [],
        "subtensor_mechanism": [],
        "text": [],
    }
    
    for idx in range(len(filled_strings)):
        save_path = pathlib.Path(f"{prompt_2_output_save_to_prefix}_idx_{idx}.json")

        level_of_difficulty = filled_strings[idx].values["level_of_difficulty"]
        tensor_name = filled_strings[idx].values["tensor_name"]
        source_domain = filled_strings[idx].values["source_domain"]
        subtensor_name = filled_strings[idx].values["subtensor_name"]
        subtensor_definition = filled_strings[idx].values["subtensor_definition"]
        
        if save_path.exists():
            print(f"Loading from {save_path}")
            out_content = load_json(save_path)
        else:
            # call model
            out = Completion(filled_strings[idx].filled, is_json=True)
            out_content = json.loads(out.choices[idx].message.content)
            
            # save output
            save_json(out_content, save_path)
            print(f"Called model and saved to {save_path}")
        
        pre_df["level_of_difficulty"].append(level_of_difficulty)
        pre_df["tensor_name"].append(tensor_name)
        pre_df["source_domain"].append(source_domain)
        pre_df["subtensor_name"].append(subtensor_name)
        pre_df["subtensor_definition"].append(subtensor_definition)
        # new
        pre_df["subtensor_purpose"].append(out_content["purpose"])
        pre_df["subtensor_mechanism"].append(out_content["mechanism"])
        pre_df["text"].append(out_content["purpose"] + " " + out_content["mechanism"])

    df_prompt_3_input = pd.DataFrame(pre_df)
    df_prompt_3_input.to_csv(prompt_2_output_save_to)
    
    return df_prompt_3_input


def redact_func(str, annotated_str, list_of_redact_words):
    """
    Replace everything inside {{}} with XXX.
    Replace all instances of {{}} with XXX.
    Replace all instances of match_words with XXX
    """    
    # Add annotated words
    annotated_words = [match.replace("{{", "").replace("}}", "") for match in re.findall(r'\{\{[^\}]*\}\}', annotated_str)]

    # Add capitalized words not following punctuation
    capitalized_words = [match.strip() for match in re.findall(r'\.? *[A-Z][a-z]*', str) if 
                         not match.startswith(".") and 
                         not match.startswith("?") and
                         not match.startswith("!")
                        ]
    
    # Add other known words
    redact_words = [*annotated_words, *capitalized_words, *list_of_redact_words]

    # Split on spaces
    nested_list = [word.split(" ") for word in redact_words]
    flat_list = []
    for sublist in nested_list:
        for item in sublist:
            flat_list.append(item)
    redact_words = flat_list
    
    # Handle singular versions
    singular_words = [word[:-1] for word in redact_words if word.endswith("s")]
    redact_words = [*redact_words, *singular_words]
    
    # Redact
    str_text = str
    for word in redact_words:
        str_text = str_text.replace(word, "XXX")
    return str_text


def run_prompt_3(filled_strings):
    pre_df = {
        "level_of_difficulty": [],
        "tensor_name": [],
        "source_domain": [],
        "subtensor_name": [],
        "subtensor_definition": [],
        "subtensor_purpose": [],
        "subtensor_mechanism": [],
        "text": [],
        "text_annotated": [],
        "text_redacted": [],
    }
    
    for idx in range(len(filled_strings)):
        save_path = pathlib.Path(f"{prompt_3_output_save_to_prefix}_idx_{idx}.txt")

        level_of_difficulty = filled_strings[idx].values["level_of_difficulty"]
        tensor_name = filled_strings[idx].values["tensor_name"]
        source_domain = filled_strings[idx].values["source_domain"]
        subtensor_name = filled_strings[idx].values["subtensor_name"]
        subtensor_definition = filled_strings[idx].values["subtensor_definition"]
        subtensor_purpose = filled_strings[idx].values["subtensor_purpose"]
        subtensor_mechanism = filled_strings[idx].values["subtensor_mechanism"]
        text = filled_strings[idx].values["text"]
        
        if save_path.exists():
            print(f"Loading from {save_path}")
            with open(save_path, 'r') as file:
                out_content = file.read()
        else:
            # call model
            out = Completion(filled_strings[idx].filled, is_json=False)
            out_content = out.choices[idx].message.content
            
            # save output
            with open(save_path, 'w+') as file:
                file.write(out_content)
            print(f"Called model and saved to {save_path}")
        
        pre_df["level_of_difficulty"].append(level_of_difficulty)
        pre_df["tensor_name"].append(tensor_name)
        pre_df["source_domain"].append(source_domain)
        pre_df["subtensor_name"].append(subtensor_name)
        pre_df["subtensor_definition"].append(subtensor_definition)
        pre_df["subtensor_purpose"].append(subtensor_purpose)
        pre_df["subtensor_mechanism"].append(subtensor_mechanism)
        pre_df["text"].append(text)
        # new
        pre_df["text_annotated"].append(out_content)
        pre_df["text_redacted"].append(
            redact_func(
                text, 
                out_content, 
                [tensor_name, source_domain, subtensor_name]
            )
        )

    df_prompt_4_input = pd.DataFrame(pre_df)
    df_prompt_4_input.to_csv(prompt_3_output_save_to)
    
    return df_prompt_4_input


def run_prompt_4(filled_strings):
    pre_df = {
        "level_of_difficulty": [],
        "tensor_name": [],
        "source_domain": [],
        "subtensor_name": [],
        "subtensor_definition": [],
        "subtensor_purpose": [],
        "subtensor_mechanism": [],
        "text": [],
        "text_annotated": [],
        "text_redacted": [],
        "schema": [],
        "target_domain": [],
    }

    for idx in range(len(filled_strings)):
        save_path = pathlib.Path(f"{prompt_4_output_save_to_prefix}_idx_{idx}.json")

        level_of_difficulty = filled_strings[idx].values["level_of_difficulty"]
        tensor_name = filled_strings[idx].values["tensor_name"]
        source_domain = filled_strings[idx].values["source_domain"]
        subtensor_name = filled_strings[idx].values["subtensor_name"]
        subtensor_definition = filled_strings[idx].values["subtensor_definition"]
        subtensor_purpose = filled_strings[idx].values["subtensor_purpose"]
        subtensor_mechanism = filled_strings[idx].values["subtensor_mechanism"]
        text = filled_strings[idx].values["text"]
        text_annotated = filled_strings[idx].values["text_annotated"]
        text_redacted = filled_strings[idx].values["text_redacted"]
        
        if save_path.exists():
            print(f"Loading from {save_path}")
            out_content = load_json(save_path)
        else:
            # call model
            out = Completion(filled_strings[idx].filled, is_json=True)
            out_content = json.loads(out.choices[idx].message.content)
            
            # save output
            save_json(out_content, save_path)
            print(f"Called model and saved to {save_path}")

        for target_idx in range(len(target_domains)):
            pre_df["level_of_difficulty"].append(level_of_difficulty)
            pre_df["tensor_name"].append(tensor_name)
            pre_df["source_domain"].append(source_domain)
            pre_df["subtensor_name"].append(subtensor_name)
            pre_df["subtensor_definition"].append(subtensor_definition)
            pre_df["subtensor_purpose"].append(subtensor_purpose)
            pre_df["subtensor_mechanism"].append(subtensor_mechanism)
            pre_df["text"].append(text)
            pre_df["text_annotated"].append(text_annotated)
            pre_df["text_redacted"].append(text_redacted)
            # new
            pre_df["schema"].append(json.dumps(out_content["engineering_design_principles"], indent=2))
            pre_df["target_domain"].append(target_domains[target_idx])

    df_prompt_5_input = pd.DataFrame(pre_df)
    df_prompt_5_input.head()
    df_prompt_5_input.to_csv(prompt_4_output_save_to)
    
    return df_prompt_5_input


def run_prompt_5(filled_strings):
    pre_df = {
        "level_of_difficulty": [],
        "tensor_name": [],
        "source_domain": [],
        "subtensor_name": [],
        "subtensor_definition": [],
        "subtensor_purpose": [],
        "subtensor_mechanism": [],
        "text": [],
        "text_annotated": [],
        "text_redacted": [],
        "schema": [],
        "target_domain": [],
        "subvehicle_name": [],
        "subtensor_name_as_json_key": [],
        "subvehicle_name_as_json_key": [],
    }

    for idx in range(len(filled_strings)):
        save_path = pathlib.Path(f"{prompt_5_output_save_to_prefix}_idx_{idx}.txt")

        level_of_difficulty = filled_strings[idx].values["level_of_difficulty"]
        tensor_name = filled_strings[idx].values["tensor_name"]
        source_domain = filled_strings[idx].values["source_domain"]
        subtensor_name = filled_strings[idx].values["subtensor_name"]
        subtensor_definition = filled_strings[idx].values["subtensor_definition"]
        subtensor_purpose = filled_strings[idx].values["subtensor_purpose"]
        subtensor_mechanism = filled_strings[idx].values["subtensor_mechanism"]
        text = filled_strings[idx].values["text"]
        text_annotated = filled_strings[idx].values["text_annotated"]
        text_redacted = filled_strings[idx].values["text_redacted"]
        schema = filled_strings[idx].values["schema"]
        target_domain = filled_strings[idx].values["target_domain"]
        
        if save_path.exists():
            print(f"Loading from {save_path}")
            with open(save_path, 'r') as file:
                out_content = file.read()
        else:
            # call model
            out = Completion(filled_strings[idx].filled, is_json=False, is_short_ans=True)
            out_content = out.choices[idx].message.content
            
            # save output
            with open(save_path, 'w+') as file:
                file.write(out_content)
            print(f"Called model and saved to {save_path}")

        pre_df["level_of_difficulty"].append(level_of_difficulty)
        pre_df["tensor_name"].append(tensor_name)
        pre_df["source_domain"].append(source_domain)
        pre_df["subtensor_name"].append(subtensor_name)
        pre_df["subtensor_definition"].append(subtensor_definition)
        pre_df["subtensor_purpose"].append(subtensor_purpose)
        pre_df["subtensor_mechanism"].append(subtensor_mechanism)
        pre_df["text"].append(text)
        pre_df["text_annotated"].append(text_annotated)
        pre_df["text_redacted"].append(text_redacted)
        pre_df["schema"].append(schema)
        pre_df["target_domain"].append(target_domain)
        # new
        pre_df["subvehicle_name"].append(out_content)
        pre_df["subtensor_name_as_json_key"].append(subtensor_name.replace(" ", "_").lower())
        pre_df["subvehicle_name_as_json_key"].append(out_content.replace(" ", "_").lower())

    df_prompt_6_input = pd.DataFrame(pre_df)
    df_prompt_6_input.to_csv(prompt_5_output_save_to)
    
    return df_prompt_6_input


def run_prompt_6(filled_strings):
    pre_df = {
        "level_of_difficulty": [],
        "tensor_name": [],
        "source_domain": [],
        "subtensor_name": [],
        "subtensor_definition": [],
        "subtensor_purpose": [],
        "subtensor_mechanism": [],
        "text": [],
        "text_annotated": [],
        "text_redacted": [],
        "schema": [],
        "target_domain": [],
        "subvehicle_name": [],
        "subtensor_name_as_json_key": [],
        "subvehicle_name_as_json_key": [],
        "extended_metaphor": [], 
    }

    for idx in range(len(filled_strings)):
        save_path = pathlib.Path(f"{prompt_6_output_save_to_prefix}_idx_{idx}.json")

        level_of_difficulty = filled_strings[idx].values["level_of_difficulty"]
        tensor_name = filled_strings[idx].values["tensor_name"]
        source_domain = filled_strings[idx].values["source_domain"]
        subtensor_name = filled_strings[idx].values["subtensor_name"]
        subtensor_definition = filled_strings[idx].values["subtensor_definition"]
        subtensor_purpose = filled_strings[idx].values["subtensor_purpose"]
        subtensor_mechanism = filled_strings[idx].values["subtensor_mechanism"]
        text = filled_strings[idx].values["text"]
        text_annotated = filled_strings[idx].values["text_annotated"]
        text_redacted = filled_strings[idx].values["text_redacted"]
        schema = filled_strings[idx].values["schema"]
        target_domain = filled_strings[idx].values["target_domain"]
        subvehicle_name = filled_strings[idx].values["subvehicle_name"]
        subtensor_name_as_json_key = filled_strings[idx].values["subtensor_name_as_json_key"]
        subvehicle_name_as_json_key = filled_strings[idx].values["subvehicle_name_as_json_key"]
        
        if save_path.exists():
            print(f"Loading from {save_path}")
            out_content = load_json(save_path)
        else:
            # call model
            out = Completion(filled_strings[idx].filled, is_json=True)
            out_content = json.loads(out.choices[idx].message.content)
            
            # save output
            save_json(out_content, save_path)
            print(f"Called model and saved to {save_path}")

        pre_df["level_of_difficulty"].append(level_of_difficulty)
        pre_df["tensor_name"].append(tensor_name)
        pre_df["source_domain"].append(source_domain)
        pre_df["subtensor_name"].append(subtensor_name)
        pre_df["subtensor_definition"].append(subtensor_definition)
        pre_df["subtensor_purpose"].append(subtensor_purpose)
        pre_df["subtensor_mechanism"].append(subtensor_mechanism)
        pre_df["text"].append(text)
        pre_df["text_annotated"].append(text_annotated)
        pre_df["text_redacted"].append(text_redacted)
        pre_df["schema"].append(schema)
        pre_df["target_domain"].append(target_domain)
        pre_df["subvehicle_name"].append(subvehicle_name)
        pre_df["subtensor_name_as_json_key"].append(subtensor_name_as_json_key)
        pre_df["subvehicle_name_as_json_key"].append(subvehicle_name_as_json_key)
        # new
        pre_df["extended_metaphor"].append(json.dumps(out_content["extended_metaphor"], indent=2))

    df_prompt_7_input = pd.DataFrame(pre_df)
    df_prompt_7_input.to_csv(prompt_6_output_save_to)
    
    return df_prompt_7_input


def run_prompt_7(filled_strings):
    pre_df = {
        "level_of_difficulty": [],
        "tensor_name": [],
        "source_domain": [],
        "subtensor_name": [],
        "subtensor_definition": [],
        "subtensor_purpose": [],
        "subtensor_mechanism": [],
        "text": [],
        "text_annotated": [],
        "text_redacted": [],
        "schema": [],
        "target_domain": [],
        "subvehicle_name": [],
        "subtensor_name_as_json_key": [],
        "subvehicle_name_as_json_key": [],
        "extended_metaphor": [],
        "final_output": [], 
    }

    for idx in range(len(filled_strings)):
        save_path = pathlib.Path(f"{prompt_7_output_save_to_prefix}_idx_{idx}.txt")

        level_of_difficulty = filled_strings[idx].values["level_of_difficulty"]
        tensor_name = filled_strings[idx].values["tensor_name"]
        source_domain = filled_strings[idx].values["source_domain"]
        subtensor_name = filled_strings[idx].values["subtensor_name"]
        subtensor_definition = filled_strings[idx].values["subtensor_definition"]
        subtensor_purpose = filled_strings[idx].values["subtensor_purpose"]
        subtensor_mechanism = filled_strings[idx].values["subtensor_mechanism"]
        text = filled_strings[idx].values["text"]
        text_annotated = filled_strings[idx].values["text_annotated"]
        text_redacted = filled_strings[idx].values["text_redacted"]
        schema = filled_strings[idx].values["schema"]
        target_domain = filled_strings[idx].values["target_domain"]
        subvehicle_name = filled_strings[idx].values["subvehicle_name"]
        subtensor_name_as_json_key = filled_strings[idx].values["subtensor_name_as_json_key"]
        subvehicle_name_as_json_key = filled_strings[idx].values["subvehicle_name_as_json_key"]
        extended_metaphor = filled_strings[idx].values["extended_metaphor"]
        
        if save_path.exists():
            print(f"Loading from {save_path}")
            with open(save_path, 'r') as file:
                out_content = file.read()
        else:
            # call model
            out = Completion(filled_strings[idx].filled, is_json=False)
            out_content = out.choices[idx].message.content
            
            # save output
            with open(save_path, 'w+') as file:
                file.write(out_content)
            print(f"Called model and saved to {save_path}")

        pre_df["level_of_difficulty"].append(level_of_difficulty)
        pre_df["tensor_name"].append(tensor_name)
        pre_df["source_domain"].append(source_domain)
        pre_df["subtensor_name"].append(subtensor_name)
        pre_df["subtensor_definition"].append(subtensor_definition)
        pre_df["subtensor_purpose"].append(subtensor_purpose)
        pre_df["subtensor_mechanism"].append(subtensor_mechanism)
        pre_df["text"].append(text)
        pre_df["text_annotated"].append(text_annotated)
        pre_df["text_redacted"].append(text_redacted)
        pre_df["schema"].append(schema)
        pre_df["target_domain"].append(target_domain)
        pre_df["subvehicle_name"].append(subvehicle_name)
        pre_df["subtensor_name_as_json_key"].append(subtensor_name_as_json_key)
        pre_df["subvehicle_name_as_json_key"].append(subvehicle_name_as_json_key)
        pre_df["extended_metaphor"].append(extended_metaphor)
        # new
        pre_df["final_output"].append(out_content)

    df_final = pd.DataFrame(pre_df)
    df_final.to_csv(prompt_7_output_save_to)
    
    return df_final


def main():
    df_prompt_1_input = pd.read_csv(pathlib.Path("../prompt_fills/concepts_per_domain.csv"))
    df_prompt_1_input = df_prompt_1_input.rename(columns={"Scientific Domain": "source_domain", "Main Tenor": "tensor_name", "Level of Difficulty": "level_of_difficulty"})

    filled_strings = get_filled_strings_from_dataframe(prompt_1_template, df_prompt_1_input)
    df_prompt_2_input = run_prompt_1(filled_strings)

    filled_strings = get_filled_strings_from_dataframe(prompt_2_template, df_prompt_2_input)
    df_prompt_3_input = run_prompt_2(filled_strings)

    filled_strings = get_filled_strings_from_dataframe(prompt_3_template, df_prompt_3_input)
    df_prompt_4_input = run_prompt_3(filled_strings)

    filled_strings = get_filled_strings_from_dataframe(prompt_4_template, df_prompt_4_input)
    df_prompt_5_input = run_prompt_4(filled_strings)

    filled_strings = get_filled_strings_from_dataframe(prompt_5_template, df_prompt_5_input)
    df_prompt_6_input = run_prompt_5(filled_strings)

    filled_strings = get_filled_strings_from_dataframe(prompt_6_template, df_prompt_6_input)
    df_prompt_7_input = run_prompt_6(filled_strings)

    filled_strings = get_filled_strings_from_dataframe(prompt_7_template, df_prompt_7_input)
    final_output = run_prompt_7(filled_strings)

    print("Finished running script.")
    print(f"Final number of rows: {len(final_output)}")