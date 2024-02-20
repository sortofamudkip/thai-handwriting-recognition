import logging
from pathlib import Path

def create_output_dir(pipeline_name: str, skip_if_exists=False) -> Path:
    """
    Creates a new directory for the pipeline's output files.

    Args:
        pipeline_name (str): The name of the pipeline.
        skip_if_exists (bool): If True, returns the existing directory instead of creating a new one.

    Returns:
        Path: The path to the newly created directory.
    """
    output_dir = (Path(__file__).parent / "./jobs/results" / pipeline_name).resolve()
    if skip_if_exists and output_dir.is_dir():
        return output_dir
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
        logging.info(f"Created output dir: {output_dir}")
    except FileExistsError:
        logging.error(f"Dir already exists: {output_dir}")
        assert False
    return output_dir
