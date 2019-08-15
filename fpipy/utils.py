def _drop_variable(ds, variable, keep_variables):
    """Drop a given variable from the dataset unless whitelisted.

    Parameters
    ----------

    ds : xr.Dataset
        Dataset to drop variable from.

    variable : str
        Variable name to drop.

    keep_variables : list-like
        Whitelist of variables to keep.

    Returns
    -------
    xr.Dataset
        Original dataset with or without the given variable.
    """
    if not keep_variables or variable not in keep_variables:
        return ds.drop(variable)
    else:
        return ds
