import os
import modal
import random
import pandas as pd
import hopsworks
from find_min_and_max_values_for_wine import generate_n_wine_samples


LOCAL=True
VERSION=1

if LOCAL == False:
   stub = modal.Stub("wine_daily")
   image = modal.Image.debian_slim().pip_install(["hopsworks"]) 

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("hopsworks api key"))
   def f():
       g()

def g():

        project = hopsworks.login()
        fs = project.get_feature_store()

        wine_df = generate_n_wine_samples(200)
        print(wine_df.head())

        wine_fg = fs.get_feature_group(name="wine",version=VERSION)
        # replace the "quality" column from double to bigint
        wine_df['quality'] = wine_df['quality'].astype('int64')
        wine_fg.insert(wine_df)


def eda():
    project = hopsworks.login()
    fs = project.get_feature_store()
    wine_red_df = pd.read_csv("wine+quality/winequality-red.csv", sep=';')
    # add a column to indicate the red wine type
    wine_red_df['type'] = -1


    wine_white_df = pd.read_csv("wine+quality/winequality-white.csv", sep=';')
    # add a column to indicate the white wine type
    wine_white_df['type'] = 1

    # concatenate the two datasets
    wine_df = pd.concat([wine_red_df, wine_white_df], axis=0)

    # set lowercase
    wine_df.columns = wine_df.columns.str.lower()

    # replace spaces with underscores
    wine_df.columns = wine_df.columns.str.replace(' ', '_')
    wine_fg = fs.get_or_create_feature_group(
    name="wine",
    version=VERSION,
    primary_key=list(wine_df.columns.drop('type').drop('quality')),
    description="Wine quality dataset")
    wine_fg.insert(wine_df)
    from great_expectations.core import ExpectationSuite, ExpectationConfiguration

    def expect(suite, column, min_val, max_val):
        suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={
                "column":column, 
                "min_value":min_val,
                "max_value":max_val,
            }
        )
    )
    suite = ExpectationSuite(expectation_suite_name="wine_characteristics")

    import numpy as np
    from scipy import stats

    for column in wine_df.columns:
        if column in ["quality", "type"]:
            continue

        mean = wine_df[column].mean()
        std_dev = wine_df[column].std()


        # I am allowing 5 standard deviations from the mean
        # And since that still had some outliers I added 1 to each side
        lower_bound = (mean - 5 * std_dev) - 1
        upper_bound = (mean + 5 * std_dev) + 1
        # Create expectation for the column using the calculated bounds
        expect(suite, column, lower_bound, upper_bound)

    # Save the expectation suite
    wine_fg.save_expectation_suite(expectation_suite=suite, validation_ingestion_policy="STRICT")


if __name__ == "__main__":
    # replaces the eda notebook
    # eda()
    if LOCAL == True :
        g()
    else:
        stub.deploy("wine_daily")
        with stub.run():
            f()
