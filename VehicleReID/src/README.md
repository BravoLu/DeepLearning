# Vehicle ReID


# generate_train_val_list.py
model_attr.txt->dic_vehicleID_modelID:{[vehicleID, modelID]}
img2vid.txt->dic_img_vehicleID:{[image_name, vehicleID]}
[image_name,vehicleID]+[vehicleID, modelID] -> dic_modelID_imgPath{[modelID, image_name]}
if the number of the same model ID reach over 20, 90% to train and 10% to validation,else 100% to train.

* result
train_vehicleModel_list.txt
val_vehicleModel_list.txt 
[image_path, modelID]

# loss.py 
identity_loss(y_true, y_pred)?



# predict.py
1. load model from MODEL_PATH
2. f_acs_extractor - use the F_ACS-1024 layer as features.
3. f_sls3_extracor - use the F_SLS3-256 layer as features.
4. predict_generator?