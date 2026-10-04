"""The prediction engine. Pure pandas: DataFrames in, DataFrames out.

Nothing in this package imports the database, HTTP or web layers; loading data
is formcast.loaders' job.
"""

from formcast.predict.model import ModelParams, predict_points

__all__ = ["ModelParams", "predict_points"]
