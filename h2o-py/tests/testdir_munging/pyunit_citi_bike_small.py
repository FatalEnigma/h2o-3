#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Explore a typical Data Science workflow with H2O and Python.

Goal: assist the manager of CitiBike of NYC to load-balance the bicycles
across the CitiBike network of stations, by predicting the number of bike
trips taken from the station every day.  Use 10 million rows of historical
data, and eventually add weather data.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import time

import h2o
from h2o.estimators import (H2ODeepLearningEstimator, H2OGeneralizedLinearEstimator, H2OGradientBoostingEstimator,
                            H2ORandomForestEstimator)
from h2o.utils.shared_utils import _locate  # private function. used to find files within h2o git project directory.
from tests import pyunit_utils


def test_0():
    """Copy of pyunit_citi_bike_small.py."""
    # 1- Load data - 1 row per bicycle trip.  Has columns showing the start and end
    # station, trip duration and trip start time and day.  The larger dataset
    # totals about 10 million rows.
    print("Import and Parse bike data")
    data = h2o.import_file(_locate("bigdata/laptop/citibike-nyc/2013-10.csv"))

    # 2- light data munging: group the bike starts per-day, converting the 10M rows
    # of trips to about 140,000 station & day combos - predicting the number of trip
    # starts per-station-per-day.

    # Convert start time to: Day since the Epoch
    startime = data["starttime"]
    secs_per_day = 1000 * 60 * 60 * 24
    data["Days"] = (startime / secs_per_day).floor()
    data.describe()

    # Now do a monster Group-By.  Count bike starts per-station per-day.  Ends up
    # with about 340 stations times 400 days (140,000 rows).  This is what we want
    # to predict.
    grouped = data.group_by(["Days", "start station name"])
    bpd = grouped.count().get_frame()  # Compute bikes-per-day
    bpd.set_name(2, "bikes")
    bpd.show()
    bpd.describe()
    print(bpd.dim)

    # Quantiles: the data is fairly unbalanced; some station/day combos are wildly
    # more popular than others.
    print("Quantiles of bikes-per-day")
    bpd["bikes"].quantile().show()

    # A little feature engineering
    # Add in month-of-year (seasonality; fewer bike rides in winter than summer)
    secs = bpd["Days"] * secs_per_day
    bpd["Month"] = secs.month().asfactor()
    # Add in day-of-week (work-week; more bike rides on Sunday than Monday)
    bpd["DayOfWeek"] = secs.dayOfWeek()

    # ----------
    # 3- Fit a model on train; using test as validation

    # Function for doing class test/train/holdout split
    def split_fit_predict(data):
        # Classic Test/Train split
        r = data["Days"].runif()   # Random uniform numbers, one per row
        train = data[r < 0.6]
        test = data[(0.6 <= r) & (r < 0.9)]
        hold = data[0.9 <= r]
        print("Training data has %d columns and %d rows, test has %d rows, holdout has %d"
              % (train.ncol, train.nrow, test.nrow, hold.nrow))
        bike_names_x = list(data.names)  # make a copy of the array
        bike_names_x.remove("bikes")

        # Run GBM
        s = time.time()
        gbm0 = H2OGradientBoostingEstimator(ntrees=500, max_depth=6, learn_rate=0.1)
        gbm0.train(x=bike_names_x, y="bikes", training_frame=train, validation_frame=test)
        gbm0_elapsed = time.time() - s

        # Run DRF
        s = time.time()
        drf0 = H2ORandomForestEstimator(ntrees=250, max_depth=30)
        drf0.train(x=bike_names_x, y="bikes", training_frame=train, validation_frame=test)
        drf0_elapsed = time.time() - s

        # Run GLM
        if "WC1" in bike_names_x: bike_names_x.remove("WC1")
        s = time.time()
        glm0 = H2OGeneralizedLinearEstimator(Lambda=[1e-5], family="poisson")
        glm0.train(x=bike_names_x, y="bikes", training_frame=train, validation_frame=test)
        glm0_elapsed = time.time() - s

        # Run DL
        s = time.time()
        dl0 = H2ODeepLearningEstimator(hidden=[50, 50, 50, 50], epochs=50)
        dl0.train(x=bike_names_x, y="bikes", training_frame=train, validation_frame=test)
        dl0_elapsed = time.time() - s

        # ----------
        # 4- Score on holdout set & report
        train_mse_gbm = gbm0.model_performance(train).mse()
        test_mse_gbm = gbm0.model_performance(test).mse()
        hold_mse_gbm = gbm0.model_performance(hold).mse()

        train_mse_drf = drf0.model_performance(train).mse()
        test_mse_drf = drf0.model_performance(test).mse()
        hold_mse_drf = drf0.model_performance(hold).mse()

        train_mse_glm = glm0.model_performance(train).mse()
        test_mse_glm = glm0.model_performance(test).mse()
        hold_mse_glm = glm0.model_performance(hold).mse()

        train_mse_dl = dl0.model_performance(train).mse()
        test_mse_dl = dl0.model_performance(test).mse()
        hold_mse_dl = dl0.model_performance(hold).mse()

        # make a pretty HTML table printout of the results
        header = ["Model", "mse TRAIN", "mse TEST", "mse HOLDOUT", "Model Training Time (s)"]
        table = [["GBM", train_mse_gbm, test_mse_gbm, hold_mse_gbm, round(gbm0_elapsed, 3)],
                 ["DRF", train_mse_drf, test_mse_drf, hold_mse_drf, round(drf0_elapsed, 3)],
                 ["GLM", train_mse_glm, test_mse_glm, hold_mse_glm, round(glm0_elapsed, 3)],
                 ["DL",  train_mse_dl,  test_mse_dl,  hold_mse_dl,  round(dl0_elapsed, 3)]]
        h2o.display.H2ODisplay(table, header)
        # --------------

    # Split the data (into test & train), fit some models and predict on the holdout data
    split_fit_predict(bpd)

    # Here we see an r^2 of 0.91 for GBM, and 0.71 for GLM.  This means given just
    # the station, the month, and the day-of-week we can predict 90% of the
    # variance of the bike-trip-starts.
    # ----------
    # 5- Now lets add some weather
    # Load weather data
    wthr1 = h2o.import_file(path=[_locate("bigdata/laptop/citibike-nyc/31081_New_York_City__Hourly_2013.csv"),
                                  _locate("bigdata/laptop/citibike-nyc/31081_New_York_City__Hourly_2014.csv")])
    # Peek at the data
    wthr1.describe()
    # Lots of columns in there!  Lets plan on converting to time-since-epoch to do
    # a "join" with the bike data, plus gather weather info that might affect
    # cyclists - rain, snow, temperature.  Alas, drop the "snow" column since it's
    # all NA's.  Also add in dew point and humidity just in case.  Slice out just
    # the columns of interest and drop the rest.
    wthr2 = wthr1[["Year Local", "Month Local", "Day Local", "Hour Local", "Dew Point (C)",
                   "Humidity Fraction", "Precipitation One Hour (mm)", "Temperature (C)",
                   "Weather Code 1/ Description"]]

    wthr2.set_name(wthr2.names.index("Precipitation One Hour (mm)"), "Rain (mm)")
    wthr2.set_name(wthr2.names.index("Weather Code 1/ Description"), "WC1")
    wthr2.describe()

    # Much better!
    # Filter down to the weather at Noon
    wthr3 = wthr2[wthr2["Hour Local"] == 12]
    # Lets now get Days since the epoch... we'll convert year/month/day into Epoch
    # time, and then back to Epoch days.  Need zero-based month and days, but have
    # 1-based.
    wthr3["msec"] = h2o.H2OFrame.mktime(year=wthr3["Year Local"], month=wthr3["Month Local"] - 1,
                                        day=wthr3["Day Local"] - 1, hour=wthr3["Hour Local"])
    secs_per_day = 1000 * 60 * 60 * 24
    wthr3["Days"] = (wthr3["msec"] / secs_per_day).floor()
    wthr3.describe()

    # msec looks sane (numbers like 1.3e12 are in the correct range for msec since
    # 1970).  Epoch Days matches closely with the epoch day numbers from the
    # CitiBike dataset.
    # Lets drop off the extra time columns to make a easy-to-handle dataset.
    wthr4 = wthr3.drop("Year Local").drop("Month Local").drop("Day Local").drop("Hour Local").drop("msec")

    # Also, most rain numbers are missing - lets assume those are zero rain days
    rain = wthr4["Rain (mm)"]
    rain[rain.isna()] = 0
    wthr4["Rain (mm)"] = rain

    # ----------
    # 6 - Join the weather data-per-day to the bike-starts-per-day
    print(bpd)
    print(wthr4)
    print("Merge Daily Weather with Bikes-Per-Day")
    bpd_with_weather = bpd.merge(wthr4, all_x=True, all_y=False)
    bpd_with_weather.describe()
    bpd_with_weather.show()

    # 7 - Test/Train split again, model build again, this time with weather
    split_fit_predict(bpd_with_weather)


if __name__ == "__main__":
    pyunit_utils.standalone_test(test_0)
else:
    test_0()
