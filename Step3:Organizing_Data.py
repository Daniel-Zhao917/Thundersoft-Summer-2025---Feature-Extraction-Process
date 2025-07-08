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

# This function will extract the seven required attributes
# Note: "P-scale" is calculated as the mean of "Pupil_Scale (Left)" and "Pupil_Scale (Right)"
# We didn't rename all the attribute to match the name given on the spec. For example, the "gaze_x" attribute on the spec is named as "gaze_angle_x" here
def preprocessing(data):
    return data[ ["gaze_angle_x", "gaze_angle_y", "EAR", "pose_Rx", "pose_Ry", "pose_Rz", "P-scale"] ]


# path is the path that doesn't contain the csv name nor the "/" before the csv name, for example, "Step2Deliverable," 
# or "/Users/stevenwang/Desktop/日后可能会有用的文件/BAC Classification/Step2Deliverables"

# file name list is a iteratable container that consist the names of the people, for example, a list of ["19DA666D", "4B725FCB_0"]

# The premise is that we assume all the csv files are stored in a same folder. We also assume that the structure of the csv files
# are 6 csv files per person, each person has 3 BAC levels, and then 2 channels. So for a person with name "Yuelin," the csv files are
# "CHO1_Yuelin_0.csv," "CHO1_Yuelin_5.csv," "CHO1_Yuelin_10.csv," "CHO2_Yuelin_0.csv," "CHO2_Yuelin_5.csv," and "CHO2_Yuelin_10.csv"

# Will return a series with MultiIndex. If only channel 2, then the index is 2D, first dimension is people name, second is BAC level in ["0", "5", "10"].
# Please note that BAC level is string, not numeric. For example, result["19DA666D"]["0"] will return the DataFrame of channel 2 of BAC 0 of "19DA666D"  
# If only channel 2 is false, then the index is 3D, first dimension is people name, second is BAC level, third is channel in ["CH01", "CH02"]
# For example, result["19DA666D"]["0"]["CH01"] will return the DataFrame of channel 1 of BAC 0 of "19DA666D"
def overall_normalize(path, file_name_list, only_channel_2):
    result_list = []
    channel_list = ["CH02"]
    if not only_channel_2:
        channel_list.append("CH01")
    for x in file_name_list:
        for BAC in ["0", "5", "10"]:    
            for chapter in channel_list:    
                file_path = path + "/" + chapter + "_" + x + "_" + BAC + ".csv"
                result_list.append(preprocessing(DataFrame_normalize(pd.read_csv(file_path))))
    if only_channel_2:
        series_index = pd.MultiIndex.from_product([file_name_list, ["0", "5", "10"]], names = ["people name", "BAC level"])
    else:
        series_index = pd.MultiIndex.from_product([file_name_list, ["0", "5", "10"], ["CH01", "CH02"]], names = ["people name", "BAC level", "channel"])
    series = pd.Series(result_list, index = series_index)
    return series


# %%

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

# series is a sequential container, if margin is 0.05, then when any times of 5% is completed (like 10% and 75%),
# will print "x% complete"
def print_progress(series, margin, progress):
    if len(series) < 6:
        pass
    percent = 0
    while (percent < 1):
        percent = round(percent + margin, 2)
        if progress == int(len(series) * percent):
            print(f"{ int(percent * 100) }% complete, {progress}/{len(series)}")
            return percent

# series is basically the MultiIndex series from the overall normalize function. Window series and label series
# both have the same MultiIndex as the "series" argument. Window series stores the windows, and label series stores the labels
def dataset_deliver(series):
    windows = []
    labels = []
    BAC_level = 0
    progress = 0
    for table in series:
        windows.append(sliding_window_list(table))
        labels.append(label_list(table.shape[0], BAC_level % 3))
        BAC_level = BAC_level + 1
        progress = progress + 1
        print_progress(series, 0.1, progress) # prints when times of 10% is completed
    window_series = pd.Series(windows, index = series.index)
    label_series = pd.Series(labels, index = series.index)
    return (window_series, label_series)

# %%

# %%
