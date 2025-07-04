# %%
import pandas as pd



# %%

# return the z score of the series
def z_score_normalize(series):
    return( (series - series.mean()) / series.std() )

# data is the raw data (after pandas's "read_csv")
# calculate the P-scale column and normalize the seven attribute
def DataFrame_normalize(data):
    data["gaze_angle_x"] = z_score_normalize(data["gaze_angle_x"])
    data["gaze_angle_y"] = z_score_normalize(data["gaze_angle_y"])
    data["EAR"] = z_score_normalize(data["EAR"])
    data["pose_Rx"] = z_score_normalize(data["pose_Rx"])
    data["pose_Ry"] = z_score_normalize(data["pose_Ry"])
    data["pose_Rz"] = z_score_normalize(data["pose_Rz"])
    data["P-scale"] = (data["Pupil_Scale (Left)"] + data["Pupil_Scale (Right)"]) / 2
    data["P-scale"] = z_score_normalize(data["P-scale"])
    return data

# %%
CH01_19DA666D_0_preprocessing = DataFrame_normalize(pd.read_csv("Step2Deliverables/CH01_19DA666D_0.csv"))
CH01_19DA666D_5_preprocessing = DataFrame_normalize(pd.read_csv("Step2Deliverables/CH01_19DA666D_5.csv"))
CH01_19DA666D_10_preprocessing = DataFrame_normalize(pd.read_csv("Step2Deliverables/CH01_19DA666D_10.csv"))
CH01_4B725FCB_0_preprocessing = DataFrame_normalize(pd.read_csv("Step2Deliverables/CH01_4B725FCB_0.csv"))
CH01_4B725FCB_5_preprocessing = DataFrame_normalize(pd.read_csv("Step2Deliverables/CH01_4B725FCB_5.csv"))
CH01_4B725FCB_10_preprocessing = DataFrame_normalize(pd.read_csv("Step2Deliverables/CH01_4B725FCB_10.csv"))
CH02_19DA666D_0_preprocessing = DataFrame_normalize(pd.read_csv("Step2Deliverables/CH02_19DA666D_0.csv"))
CH02_19DA666D_5_preprocessing = DataFrame_normalize(pd.read_csv("Step2Deliverables/CH02_19DA666D_5.csv"))
CH02_19DA666D_10_preprocessing = DataFrame_normalize(pd.read_csv("Step2Deliverables/CH02_19DA666D_10.csv"))
CH02_4B725FCB_0_preprocessing = DataFrame_normalize(pd.read_csv("Step2Deliverables/CH02_4B725FCB_0.csv"))
CH02_4B725FCB_5_preprocessing = DataFrame_normalize(pd.read_csv("Step2Deliverables/CH02_4B725FCB_5.csv"))
CH02_4B725FCB_10_preprocessing = DataFrame_normalize(pd.read_csv("Step2Deliverables/CH02_4B725FCB_10.csv"))

# %%

# This function will extract the seven required attributes
# Note: "P-scale" is calculated as the mean of "Pupil_Scale (Left)" and "Pupil_Scale (Right)"
# We didn't rename all the attribute to match the name given on the spec. For example, the "gaze_x" attribute on the spec is named as "gaze_angle_x" here
def preprocessing(data):
    return data[ ["gaze_angle_x", "gaze_angle_y", "EAR", "pose_Rx", "pose_Ry", "pose_Rz", "P-scale"] ]

# data should be a pandas DataFrame
# Will return a 3D list, the dimension is (total rows - 149) × 150 × 7
# 3D list[0] will return the first window, a 150 × 7 2D list
# For the 2D list, 2D list[0] will return a list of 7 attributes, the first row of this window.
# Thus, for the 2D list, the list is row oriented, not the pandas style column oriented (in another word,
# 2D list[0] will not return the first column, but will return the first row)
def sliding_window_list(data):
    window = data.iloc[:150].values.tolist()
    result = []
    result.append(window.copy()) # because of the infamous mutability property of Python, 
                                 # it's important to append a copy of a list. If not a copy,
                                 # and the "window" list change, then the "result" list will also change
    for i in range(data.shape[0] - 150):
        window.pop(0)
        window.append(data.iloc[150 + i].to_list())
        result.append(window.copy())
    return result

# return a list with the same length as the amount of the windows, the element is all the same, which is the label
def label_list(length, label):
    result = []
    for i in range(length - 150 + 1):
        result.append(label)
    return result

# data is a normalized pandas DataFrame (after the function "DataFrame_normalize")
# returns a tuple, the first element is the window numbers × 150 × 7 dataset, the last element is the label
def dataset_deliver(data, label):
    overall_data = preprocessing(data)
    return (sliding_window_list(overall_data), label_list(overall_data.shape[0], label))

# %%
CH01_19DA666D_0, CH01_19DA666D_0_label = dataset_deliver(CH01_19DA666D_0_preprocessing, 0)
CH01_19DA666D_5, CH01_19DA666D_5_label = dataset_deliver(CH01_19DA666D_5_preprocessing, 1)
CH01_19DA666D_10, CH01_19DA666D_10_label = dataset_deliver(CH01_19DA666D_10_preprocessing, 2)
CH01_4B725FCB_0, CH01_4B725FCB_0_label = dataset_deliver(CH01_4B725FCB_0_preprocessing, 0)
CH01_4B725FCB_5, CH01_4B725FCB_5_label = dataset_deliver(CH01_4B725FCB_5_preprocessing, 1)
CH01_4B725FCB_10, CH01_4B725FCB_10_label = dataset_deliver(CH01_4B725FCB_10_preprocessing, 2)
CH02_19DA666D_0, CH02_19DA666D_0_label = dataset_deliver(CH02_19DA666D_0_preprocessing, 0)
CH02_19DA666D_5, CH02_19DA666D_5_label = dataset_deliver(CH02_19DA666D_5_preprocessing, 1)
CH02_19DA666D_10, CH02_19DA666D_10_label = dataset_deliver(CH02_19DA666D_10_preprocessing, 2)
CH02_4B725FCB_0, CH02_4B725FCB_0_label = dataset_deliver(CH02_4B725FCB_0_preprocessing, 0)
CH02_4B725FCB_5, CH02_4B725FCB_5_label = dataset_deliver(CH02_4B725FCB_5_preprocessing, 1)
CH02_4B725FCB_10, CH02_4B725FCB_10_label = dataset_deliver(CH02_4B725FCB_10_preprocessing, 2)

# %%




# %%


