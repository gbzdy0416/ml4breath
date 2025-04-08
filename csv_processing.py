import pandas as pd
import numpy as np
import argparse


def parse_and_average(value):
    try:
        parts = value.split('_')
        nums = list(map(float, parts))
        return sum(nums) / len(nums)
    except:
        return np.nan


def process_csv(input_file, output_file):
    df = pd.read_csv(input_file, header=0, skiprows=1)
    df_subset = df.iloc[:, 12:22].copy()
    df_subset.iloc[:, 7] = df.iloc[:, 19].apply(parse_and_average)
    df_subset = df_subset[df_subset.iloc[:, 7] >= 2.5]
    df_subset = df_subset[df_subset.iloc[:, 5] < df_subset.iloc[:, 0]]
    df_subset.iloc[:, 5] = -1 * (df_subset.iloc[:, 5] - df_subset.iloc[:, 0])
    sin_factor = np.sin(df_subset.iloc[:, 6] / 100)
    multiplier = 1 + sin_factor

    df_subset.iloc[:, 2] = df_subset.iloc[:, 2].astype(float)
    df_subset.iloc[:, 2] = (df_subset.iloc[:, 2] * multiplier).round(3)

    df_subset.iloc[:, 3] = df_subset.iloc[:, 3].astype(float)
    df_subset.iloc[:, 3] = (df_subset.iloc[:, 3] * multiplier).round(3)

    df_subset.iloc[:, 4] = df_subset.iloc[:, 4].astype(float)
    df_subset.iloc[:, 4] = (df_subset.iloc[:, 4] * multiplier).round().astype(int)

    df_subset.drop(df_subset.columns[6], axis=1, inplace=True)

    df_subset.to_csv(output_file, index=False, header=False)
def main():
    process_csv("original.csv", "data.csv")


if __name__ == '__main__':
    main()