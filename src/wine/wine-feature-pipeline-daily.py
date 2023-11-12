import os
import modal
import random
import pandas as pd
import hopsworks


LOCAL=True

if LOCAL == False:
   stub = modal.Stub("wine_daily")
   image = modal.Image.debian_slim().pip_install(["hopsworks"]) 

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("hopsworks api key"))
   def f():
       g()


def generate_flower(name, sepal_len_max, sepal_len_min, sepal_width_max, sepal_width_min, 
                    petal_len_max, petal_len_min, petal_width_max, petal_width_min):
    """
    Returns a single iris flower as a single row in a DataFrame
    """
    df = pd.DataFrame({ "sepal_length": [random.uniform(sepal_len_max, sepal_len_min)],
                       "sepal_width": [random.uniform(sepal_width_max, sepal_width_min)],
                       "petal_length": [random.uniform(petal_len_max, petal_len_min)],
                       "petal_width": [random.uniform(petal_width_max, petal_width_min)]
                      })
    df['variety'] = name
    return df


def generate_wine_sample(min_values, max_values, wine_type):
    wine_sample = {}
    for feature in min_values.keys():
        wine_sample[feature] = [random.uniform(min_values[feature], max_values[feature])]

    df = pd.DataFrame(wine_sample)
    df['type'] = wine_type
    return df

def generate_n_wine_samples(min_values, max_values, wine_type, n):
    df = pd.DataFrame()
    for i in range(n):
        df = df._append(generate_wine_sample(min_values, max_values, wine_type))
    return df



def g():

    #project = hopsworks.login()
    #fs = project.get_feature_store()
    # input
    min_values_white_wine = {
        "fixed acidity": 3.8,
        "volatile acidity": 0.08,
        "citric acid": 0.0,
        "residual sugar": 0.6,
        "chlorides": 0.009,
        "free sulfur dioxide": 2.0,
        "total sulfur dioxide": 9.0,
        "density": 0.98711,
        "pH": 2.72,
        "sulphates": 0.22,
        "alcohol": 8.0,
        "quality": 3.0
    }

    max_values_white_wine = {
        "fixed acidity": 14.2,
        "volatile acidity": 1.1,
        "citric acid": 1.66,
        "residual sugar": 65.8,
        "chlorides": 0.346,
        "free sulfur dioxide": 289.0,
        "total sulfur dioxide": 440.0,
        "density": 1.03898,
        "pH": 3.82,
        "sulphates": 1.08,
        "alcohol": 14.2,
        "quality": 9.0
    }

    # Given values for red wine min and max
    min_values_red_wine = {
        "fixed acidity": 4.6,
        "volatile acidity": 0.12,
        "citric acid": 0.0,
        "residual sugar": 0.9,
        "chlorides": 0.012,
        "free sulfur dioxide": 1.0,
        "total sulfur dioxide": 6.0,
        "density": 0.99007,
        "pH": 2.74,
        "sulphates": 0.33,
        "alcohol": 8.4,
        "quality": 3.0
    }

    max_values_red_wine = {
        "fixed acidity": 15.9,
        "volatile acidity": 1.58,
        "citric acid": 1.0,
        "residual sugar": 15.5,
        "chlorides": 0.611,
        "free sulfur dioxide": 72.0,
        "total sulfur dioxide": 289.0,
        "density": 1.00369,
        "pH": 4.01,
        "sulphates": 2.0,
        "alcohol": 14.9,
        "quality": 8.0
    }
    red_wine_df = generate_n_wine_samples(min_values_red_wine, max_values_red_wine, "red", 100)
    white_wine_df = generate_n_wine_samples(min_values_white_wine, max_values_white_wine, "white", 100)
    print("Red wine sample: ")
    print(red_wine_df.head())
    print("White wine sample: ")
    print(white_wine_df.head())
    print()

    #iris_fg = fs.get_feature_group(name="iris",version=1)
    #iris_fg.insert(iris_df)

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        stub.deploy("iris_daily")
        with stub.run():
            f()
