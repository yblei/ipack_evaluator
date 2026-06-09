# convert the answer format of surveylegend to the format used in the first survey
import pandas as pd
from pathlib import Path

# read the survey legend data
input_file = Path(__file__).parent / "responses_survey_2.xlsx"

df = pd.read_excel(input_file)


cols = df.columns[df.columns.str.contains("In what order", na=False)]
subset = df.loc[:, cols]
subset.rename(columns=lambda x: x.replace("In what order would you load these objects into a shopping box?Please rank them from 1 (placed lowest, e.g. a heavy water bottle) to 14 (placed highest, e.g. a pack of fragile noodles).", ""), inplace=True)

# remove paranthesis around column names
subset.rename(columns=lambda x: x.replace(" (", "").replace(")", "").strip(), inplace=True)

# only choose rows without NaN values
subset = subset.dropna()

# cast to int
subset = subset.astype(int)

print(subset)

# print dataset stats
print(f"Number of valid submissions: {subset.shape[0]}")

participant_ids = df.loc[subset.index, "Participant ID"].rename("participant_id")

sequence_col = subset.apply(
    lambda row: ",".join(row.sort_values().index),
    axis=1,
).rename("loading_sequence")

metadata_cols = [
    col
    for col in df.columns
    if col not in subset.columns and col not in {"Participant ID", "External ID", "Warnings", "Browser", "Device"}
]
metadata = df.loc[subset.index, metadata_cols]

export_df = pd.concat([participant_ids, sequence_col], axis=1)
print(export_df)

# export to csv
output_file = Path(__file__).parent / "imported_survey_2.csv"
export_df.to_csv(output_file, index=False)