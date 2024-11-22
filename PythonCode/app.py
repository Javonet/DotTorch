# app.py
from scripts.test_model import TestModel
from scripts.test_custom_image import TestCustomImage
from scripts.train import Train

# Train().train("./models") 
# TestModel().test("./models")
result = TestCustomImage().test("./models", "./images/test_image.jpg")
print(result)