# Template Args List

## Prompt 1 - Get Sub-Tensors

### Input:
* tensor_name: subject of the metaphor
* source_domain: domain {tensor_name} is from

### Output:
* json list of objects
    * subtensor_name
    * subtensor_definition

## Prompt 2 - Get Purpose and Mechanism

### Input:
* source_domain: domain {tensor} is from
* subtensor_name
* subtensor_definition 

### Output:
* json object
    * subtensor_purpose
    * subtensor_mechanism

## Prompt 3 - Get Annotated Text

### Input:
* source_domain: domain {tensor} is from
* text: subtensor_purpose + subtensor_mechanism

### Output:
* text

## Action A - Regex to Redact Superficial Information
* replace all mentions of annotated words, source_domain, tensor_name, subtensor_name with XXX

## Prompt 4 - Create Schema

### Input:
* text: subtensor_purpose_redacted + subtensor_mechanism_redacted

### Output:
* schema: json list of objects
    * design_principle_name
    * design_principle_spec

## Prompt 5 - Identify Sub-Vehicle

### Input:
* target_domain: domain metaphor is to
* schema: json list of objects
    * design_principle_name
    * design_principle_spec

### Output:
* subvehicle_name: vehicle of the metaphor

## Action B - Regex to Redact Superficial Information
* edit subtensor_name to be valid subtensor_name_as_json_key
* edit subvehicle_name --> to be valid subvehicle_name_as_json_key

## Prompt 6 - Write Analogy

* tensor_name
* subtensor_name
* target_domain
* subvehicle_name
* schema: json list of objects
    * design_principle_name
    * design_principle_spec
* subtensor_name_as_json_key
* subvehicle_name_as_json_key

### Output:
* json list of objects
    * analogy
    * subtensor_name_as_json_key
    * subvehicle_name_as_json_key




