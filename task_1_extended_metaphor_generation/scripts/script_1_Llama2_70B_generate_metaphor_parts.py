# Script to generate schema and subvehicle metaphors with Llama2 70B Chat hosted on Replicate

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

MAX_NUMBER_API_CALLS = 20

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
# Target Domains #
##################

target_domains = ["Kpop", "baseball", "Iron Man", "money"]

##################
# Save Locations #
##################

# Save every call
prompt_1_output_save_to_prefix = pathlib.Path("../generations/script_run_Llama70B/prompt_1_output")
prompt_2_output_save_to_prefix = pathlib.Path("../generations/script_run_Llama70B/prompt_2_output")
prompt_3_output_save_to_prefix = pathlib.Path("../generations/script_run_Llama70B/prompt_3_output")
prompt_4_output_save_to_prefix = pathlib.Path("../generations/script_run_Llama70B/prompt_4_output")
prompt_5_output_save_to_prefix = pathlib.Path("../generations/script_run_Llama70B/prompt_5_output")
prompt_6_output_save_to_prefix = pathlib.Path("../generations/script_run_Llama70B/prompt_6_output")
prompt_7_output_save_to_prefix = pathlib.Path("../generations/script_run_Llama70B/prompt_7_output")


# Consolidated to csv
prompt_1_output_save_to = pathlib.Path("../generations/script_run_Llama70B/prompt_1_output.csv")
prompt_2_output_save_to = pathlib.Path("../generations/script_run_Llama70B/prompt_2_output.csv")
prompt_3_output_save_to = pathlib.Path("../generations/script_run_Llama70B/prompt_3_output.csv")
prompt_4_output_save_to = pathlib.Path("../generations/script_run_Llama70B/prompt_4_output.csv")
prompt_5_output_save_to = pathlib.Path("../generations/script_run_Llama70B/prompt_5_output.csv")
prompt_6_output_save_to = pathlib.Path("../generations/script_run_Llama70B/prompt_6_output.csv")
prompt_7_output_save_to = pathlib.Path("../generations/script_run_Llama70B/prompt_7_output.csv")


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


def fix_JSON(json_message=None):
    """Sometimes the returned output illegally escapes certain characters."""
    result = None
    try:        
        result = json.loads(json_message)
    except Exception as e:
        print("in fix_json I see Exception")
        print(json_message)
        print(e)      
        # Find the offending character index:
        idx_to_replace = int(str(e).split(' ')[-1].replace(')', ''))        
        # Remove the offending character:
        json_message = list(json_message)
        json_message[idx_to_replace] = ''
        new_message = ''.join(json_message)     
        return fix_JSON(json_message=new_message)
    return result


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
            num_calls = 0
            out_content = None
            while (num_calls < MAX_NUMBER_API_CALLS):
                try:
                    # call model
                    out = Completion(filled_strings[idx].filled, is_json=True)
                    out_content = fix_JSON(out)
                    break
                except Exception as e:
                    num_calls = num_calls + 1
                    print(f"Prompt 1 ran into error #{num_calls}. {e}")
            
            # save output
            save_json(out_content, save_path)
            print(f"Called model and saved to {save_path}")
            
            if (num_calls == MAX_NUMBER_API_CALLS):
                raise Exception(f"Too Many Calls, look at {save_path}")
        
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
            num_calls = 0
            out_content = None
            while (num_calls < MAX_NUMBER_API_CALLS):
                try:
                    # call model
                    out = Completion(filled_strings[idx].filled, is_json=True)
                    out_content = fix_JSON(out)
                    break
                except Exception as e:
                    num_calls = num_calls + 1
                    print(f"Prompt 2 ran into error #{num_calls}. {e}")
            
            # save output
            save_json(out_content, save_path)
            print(f"Called model and saved to {save_path}")

            if (num_calls == MAX_NUMBER_API_CALLS):
                raise Exception(f"Too Many Calls, look at {save_path}")
        
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
            out_content = Completion(filled_strings[idx].filled, is_json=False)
            
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
            num_calls = 0
            out_content = None
            while (num_calls < MAX_NUMBER_API_CALLS):
                try:
                    # call model
                    out = Completion(filled_strings[idx].filled, is_json=True)
                    out_content = fix_JSON(out)
                    break
                except Exception as e:
                    num_calls = num_calls + 1
                    print(f"Prompt 4 ran into error #{num_calls}. {e}")
            
            # save output
            save_json(out_content, save_path)
            print(f"Called model and saved to {save_path}")

            if (num_calls == MAX_NUMBER_API_CALLS):
                raise Exception(f"Too Many Calls, look at {save_path}")

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
            out_content = Completion(filled_strings[idx].filled, is_json=False, is_short_ans=True)
            
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
            num_calls = 0
            out_content = None
            while (num_calls < MAX_NUMBER_API_CALLS):
                try:
                    # call model
                    out = Completion(filled_strings[idx].filled, is_json=True)
                    out_content = fix_JSON(out)
                    break
                except Exception as e:
                    num_calls = num_calls + 1
                    print(f"Prompt 6 ran into error #{num_calls}. {e}")
            
            # save output
            save_json(out_content, save_path)
            print(f"Called model and saved to {save_path}")

            if (num_calls == MAX_NUMBER_API_CALLS):
                raise Exception(f"Too Many Calls, look at {save_path}")

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
            out_content = Completion(filled_strings[idx].filled, is_json=False)
            
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


if __name__ == "__main__":
    main()