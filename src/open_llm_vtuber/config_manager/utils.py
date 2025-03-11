# config_manager/utils.py
import yaml
from pathlib import Path
from typing import Union, Dict, Any, TypeVar
from pydantic import BaseModel, ValidationError
import os
import re
import chardet
from loguru import logger

from .main import Config

T = TypeVar("T", bound=BaseModel)


def read_yaml(config_path: str) -> Dict[str, Any]:
    """
    Read the specified YAML configuration file with environment variable and config parameter substitution
    and guess encoding. Return the configuration data as a dictionary.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Configuration data as a dictionary.

    Raises:
        FileNotFoundError: If the configuration file is not found.
        IOError: If the configuration file cannot be read.
    """

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    content = load_text_file_with_guess_encoding(config_path)
    if not content:
        raise IOError(f"Failed to read configuration file: {config_path}")

    # First load the raw YAML to get the config parameters
    raw_config = yaml.safe_load(content)
    
    # Get persona_prompt if it exists
    persona_prompt = raw_config.get('character_config', {}).get('persona_prompt', '')

    # Replace environment variables
    env_pattern = re.compile(r"\$\{(\w+)\}")
    def env_replacer(match):
        env_var = match.group(1)
        return os.getenv(env_var, match.group(0))
    content = env_pattern.sub(env_replacer, content)

    # Replace config parameters
    config_pattern = re.compile(r"\$\{(persona_prompt)\}")
    def config_replacer(match):
        param_name = match.group(1)
        if param_name == 'persona_prompt':
            return persona_prompt
        return match.group(0)
    content = config_pattern.sub(config_replacer, content)

    try:
        return yaml.safe_load(content)
    except yaml.YAMLError as e:
        logger.critical(f"Error parsing YAML file: {e}")
        raise e


def validate_config(config_data: dict) -> Config:
    """
    Validate configuration data against the Config model.

    Args:
        config_data: Configuration data to validate.

    Returns:
        Validated Config object.

    Raises:
        ValidationError: If the configuration fails validation.
    """
    try:
        return Config(**config_data)
    except ValidationError as e:
        logger.critical(f"Error validating configuration: {e}")
        logger.error("Configuration data:")
        logger.error(config_data)
        raise e


def load_text_file_with_guess_encoding(file_path: str) -> str | None:
    """
    Load a text file with guessed encoding.

    Parameters:
    - file_path (str): The path to the text file.

    Returns:
    - str: The content of the text file or None if an error occurred.
    """
    encodings = ["utf-8", "utf-8-sig", "gbk", "gb2312", "ascii", "cp936"]

    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding) as file:
                return file.read()
        except UnicodeDecodeError:
            continue
    # If common encodings fail, try chardet to guess the encoding
    try:
        with open(file_path, "rb") as file:
            raw_data = file.read()
        detected = chardet.detect(raw_data)
        if detected["encoding"]:
            return raw_data.decode(detected["encoding"])
    except Exception as e:
        logger.error(f"Error detecting encoding for config file {file_path}: {e}")
    return None


def save_config(config: BaseModel, config_path: Union[str, Path]):
    """
    Saves a Pydantic model to a YAML configuration file.

    Args:
        config: The Pydantic model to save.
        config_path: Path to the YAML configuration file.
    """
    config_file = Path(config_path)
    config_data = config.model_dump(
        by_alias=True, exclude_unset=True, exclude_none=True
    )

    try:
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(
                config_data,
                f,
                allow_unicode=True,
                default_flow_style=False,
                encoding="utf-8",
                sort_keys=False,
                width=float("inf")
            )
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error writing YAML file: {e}")
    
    # 验证保存的文件是否正确
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            content = f.read()
            if not all(ord(c) < 128 or c.isprintable() for c in content):
                logger.warning("检测到配置文件中可能包含非打印字符或编码问题")
    except UnicodeError as e:
        logger.error(f"配置文件编码验证失败: {e}")


def scan_config_alts_directory(config_alts_dir: str) -> list[dict]:
    """
    Scan the config_alts directory and return a list of config information.
    Each config info contains the filename and its display name from the config.

    Parameters:
    - config_alts_dir (str): The path to the config_alts directory.

    Returns:
    - list[dict]: A list of dicts containing config info:
        - filename: The actual config file name
        - name: Display name from config, falls back to filename if not specified
    """
    config_files = []

    # Add default config first
    default_config = read_yaml("conf.yaml")
    config_files.append(
        {
            "filename": "conf.yaml",
            "name": default_config.get("character_config", {}).get(
                "conf_name", "conf.yaml"
            )
            if default_config
            else "conf.yaml",
        }
    )

    # Scan other configs
    for root, _, files in os.walk(config_alts_dir):
        for file in files:
            if file.endswith(".yaml"):
                config: dict = read_yaml(os.path.join(root, file))
                config_files.append(
                    {
                        "filename": file,
                        "name": config.get("character_config", {}).get(
                            "conf_name", file
                        )
                        if config
                        else file,
                    }
                )
    logger.debug(f"Found config files: {config_files}")
    return config_files


def scan_bg_directory() -> list[str]:
    bg_files = []
    bg_dir = "backgrounds"
    for root, _, files in os.walk(bg_dir):
        for file in files:
            if file.endswith((".jpg", ".jpeg", ".png", ".gif")):
                bg_files.append(file)
    return bg_files
