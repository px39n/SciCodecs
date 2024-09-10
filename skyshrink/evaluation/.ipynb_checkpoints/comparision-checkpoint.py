import pandas as pd
from tqdm import tqdm
from ..utils import get_directory_size,convert_bytes_to_mb,convert_speed_to_mb_per_s,format_df 

def compare_accuracy(list_pds):
    return compare_attribute(list_pds,list_pds[0].accuracy_list, True, False)


def compare_meta(list_pds):
    return format_df(compare_attribute(list_pds,["original_size","compressed_size","compression_ratio", "encoding_speed", "decoding_speed"], True, True))
 

from tqdm import tqdm
import numpy as np
import pandas as pd
from skyshrink.utils import get_directory_size,convert_bytes_to_mb,convert_speed_to_mb_per_s,format_df

def compare_attribute(list_pds, attr_list, local=True, glob=True):
    # Initialize a list to store results
    results = [] 
    pds=list_pds[0]
    var_list = (pds.glob["var_list"] if local else []) + (["glob"] if glob else [])
    
    # Iterate over each pds object in the list
    for pds in tqdm(list_pds, desc="Evaluate Meta"):
        pds.sanity_check()
        pds.update_parameter(["parameter"])
        if bool(set(attr_list) & set(pds.accuracy_list)) and not pds.report.get("accuracy"):
            pds.update_parameter(["accuracy"])
        
        # Iterate over each variable in var_list
        for var in var_list:
            var_data = getattr(pds, var)
            if var_data:
                method_name = var_data.get("method", "Unknown Method")
                
                # Create a dictionary to store this row of data
                row = {"method": method_name, "var": var, "workspace_name": pds.workspace_name}
                
                # Convert sizes from bytes to MB and speeds from bytes/s to MB/s
                for acc in attr_list:
                    value = var_data.get(acc)
                    if value is not None:
                        if acc.endswith("_size"):  # Size metrics: convert from bytes to MB
                            row[acc] = convert_bytes_to_mb(value)
                        elif acc.endswith("_speed"):  # Speed metrics: convert from bytes/s to MB/s
                            row[acc] = convert_speed_to_mb_per_s(value)
                        else:
                            row[acc] = value  # Other metrics: keep as is
                
                # Append the row to results
                results.append(row)
    
    # Convert results list to pandas DataFrame
    df = pd.DataFrame(results)
    # Replace headers to include units
    df.columns = [col.replace("_size", " (MB)").replace("_speed", " (MB/s)") for col in df.columns]
    return df


