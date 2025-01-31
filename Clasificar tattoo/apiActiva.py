from inference_sdk import InferenceHTTPClient

# create an inference client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="mlT63rhUs7OKHz2LKNMC"
)

# run inference on a local image
print(CLIENT.infer(
    "C:/Users/juan.ochoa/OneDrive - INMEL INGENIERIA SAS/Documentos/Python_Codigos/Clasificar tattoo/acuarelaPrueba.jpg"
    ,model_id="tattoo_types/4"
))