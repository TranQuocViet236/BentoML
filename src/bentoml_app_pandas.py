import bentoml
import pandas as pd
from bentoml.io import NumpyNdarray, PandasDataFrame
import numpy as np


# Load transformers and model
scaler = bentoml.sklearn.get('scaler:latest').to_runner()
pca = bentoml.sklearn.get('pca:latest').to_runner()

# Load model
model = bentoml.sklearn.get('customer_segmentation_kmeans:latest').to_runner()

# Create service with the model
service = bentoml.Service('customer_segmentation_kmeans', runners=[scaler, pca, model])


@service.api(input=PandasDataFrame(), output=NumpyNdarray())
def predict(df: pd.DataFrame) -> np.array:

    # process data
    scaled_df = pd.DataFrame([scaler.run(df)], columns=df.columns)
    processed_df = pd.DataFrame([pca.run(scaled_df)], columns= ['col1', 'col2', 'col3'])

    # predict data
    result = model.run(processed_df)

    return np.array(result)