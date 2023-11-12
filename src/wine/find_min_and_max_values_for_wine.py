import pandas as pd


def find_min_max_values(csv_file):
    df = pd.read_csv(csv_file, delimiter=';')
    # Find the min and max for each column
    min_values = df.min()
    max_values = df.max()
    return min_values, max_values


white_wine_path = 'wine+quality/winequality-white.csv'
red_wine_path = 'wine+quality/winequality-red.csv'

# Find the min and max values for white wine data
white_wine_min, white_wine_max = find_min_max_values(white_wine_path)
# Find the min and max values for red wine data
red_wine_min, red_wine_max = find_min_max_values(red_wine_path)

print("White wine min values: \n", white_wine_min)
print()
print("White wine max values: \n", white_wine_max)
print()
print("Red wine min values: \n", red_wine_min)
print()
print("Red wine max values: \n", red_wine_max)
print()

white_wine_min, white_wine_max, red_wine_min, red_wine_max

