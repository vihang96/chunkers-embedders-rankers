from typing import Any, Optional, Type, TypeVar

import yaml
from pydantic import BaseModel, ValidationError

# Define a TypeVar for generic Pydantic models
T = TypeVar('T', bound=BaseModel)


def validate_yaml_string(yaml_string: str) -> bool:
    """
    Validates if the provided string is valid YAML.

    Args:
        yaml_string: The string to validate.

    Returns:
        True if the string is valid YAML, False otherwise.
    """
    try:
        yaml.safe_load(yaml_string)
        return True
    except yaml.YAMLError:
        return False


def convert_yaml_to_pydantic(yaml_string: str, model_type: Type[T]) -> Optional[T]:
    """
    Parses any valid YAML string and validates it against the specified Pydantic model type.

    Args:
        yaml_string: The YAML string to parse.
        model_type: The Pydantic model class to validate against.

    Returns:
        An instance of the Pydantic model if parsing and validation succeed, None otherwise.
        Prints errors to the console.
    """
    try:
        data = yaml.safe_load(yaml_string)
        if data is None:
            # Handle cases where YAML is valid but represents null (e.g., empty string or "null")
            # Check if the model type allows None or has defaults that make it valid
            try:
                # Attempt validation even with None data, Pydantic might handle defaults
                return model_type.model_validate(data)
            except ValidationError:
                # If validation fails for None, treat it as an invalid input for non-optional models
                print(
                    f"Error: YAML content is null or empty, and incompatible with {model_type.__name__}."
                )
                return None

        # Validate the parsed data against the provided model type
        # Pydantic V2 uses model_validate
        return model_type.model_validate(data)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML: {e}")
        return None
    except ValidationError as e:
        print(f"Schema validation error for {model_type.__name__}: {e}")
        return None
    except Exception as e:
        # Catch other potential errors
        print(f"An unexpected error occurred during YAML to Pydantic conversion: {e}")
        return None


def convert_pydantic_to_yaml(model_instance: BaseModel, **kwargs: Any) -> str:
    """
    Converts a Pydantic model instance into a YAML string.

    Args:
        model_instance: The Pydantic model instance to convert.
        **kwargs: Additional keyword arguments to pass to yaml.dump
                  (e.g., sort_keys=False, default_flow_style=False).

    Returns:
        A YAML string representation of the model instance.
    """
    # Convert Pydantic model to a dictionary suitable for YAML serialization
    # Use model_dump for Pydantic V2, mode='json' handles complex types like Enums
    data = model_instance.model_dump(mode='json')

    # Default YAML dump settings
    dump_options = {
        'sort_keys': False,
        'default_flow_style': False,
        'allow_unicode': True,
    }
    # Update with any user-provided kwargs
    dump_options.update(kwargs)

    return yaml.safe_dump(
        data, sort_keys=False, default_flow_style=False, allow_unicode=True
    )


def convert_pydantic_to_yaml_selective(
    model_instance: BaseModel, fields: list[str], **kwargs: Any
) -> str:
    """
    Converts a Pydantic model instance into a YAML string, including only specified fields.

    Args:
        model_instance: The Pydantic model instance to convert.
        fields: A list of field names to include in the YAML output.
        **kwargs: Additional keyword arguments to pass to yaml.dump
                  (e.g., sort_keys=False, default_flow_style=False).

    Returns:
        A YAML string representation of the model instance with selected fields.
    """
    # Convert Pydantic model to a dictionary, including only specified fields
    # Use model_dump for Pydantic V2, mode='json' handles complex types like Enums
    # The `include` argument expects a set for V2, or a dict for V1.
    # Assuming Pydantic V2 as per existing code.
    if not fields:
        # If no fields are specified, dump an empty YAML or handle as an error/specific case
        # For now, let's return an empty YAML mapping
        data = {}
    else:
        data = model_instance.model_dump(include=set(fields), mode='json')

    # Default YAML dump settings
    dump_options = {
        'sort_keys': False,
        'default_flow_style': False,
        'allow_unicode': True,
    }
    # Update with any user-provided kwargs
    dump_options.update(kwargs)

    return yaml.safe_dump(
        data, sort_keys=False, default_flow_style=False, allow_unicode=True
    )
