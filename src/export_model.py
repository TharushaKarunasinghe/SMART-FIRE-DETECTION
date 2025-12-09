from micromlgen import port
import joblib
import os

# Load your trained model
model = joblib.load(os.path.join('data/models', 'decision_tree_model.pkl'))

# Export to C++ code
c_code = port(model, classmap={0:'AllClear', 1:'Watch', 2:'Caution', 3:'Warning', 4:'Emergency'})

# Save as header file
with open('arduino/fire_detection_model.h', 'w') as f:
    f.write(c_code)

print("Model exported to arduino/fire_detection_model.h")
