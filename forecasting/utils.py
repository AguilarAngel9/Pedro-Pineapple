# Util functions library.
# Authors: @THEFFTKID.

def flatten_dict(d, parent_key='', sep='_'):
    """
    Flatten a nested dictionary by combining keys with their parent keys.

    Parameters:
    - d (dict): The input dictionary.
    - parent_key (str): The parent key for recursion.
    - sep (str): The separator to use between parent and child keys.

    Returns:
    - dict: The flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
