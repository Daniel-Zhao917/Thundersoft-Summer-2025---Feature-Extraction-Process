# %%
import pandas as pd
import os


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

# Will return a series with MultiIndex, each element is a DataFrame. 
# If only channel 2, then the index is 2D, first dimension is people name, second is BAC level in ["0", "5", "10"].
# Please note that BAC level is string, not numeric. For example, result["19DA666D"]["0"] will return the DataFrame of channel 2 of BAC 0 of "19DA666D"  
# If only channel 2 is false, then the index is 3D, first dimension is people name, second is BAC level, third is channel in ["CH01", "CH02"]
# For example, result["19DA666D"]["0"]["CH01"] will return the DataFrame of channel 1 of BAC 0 of "19DA666D"
def overall_normalize(path, file_name_list, only_channel_2):
    result_list = []
    channel_list = ["CH02"]
    if not only_channel_2:
        channel_list = ["CH01", "CH02"]
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
def print_progress(series, margin, progress, tab = False):
    if len(series) < 6:
        pass
    percent = 0
    while (percent < 1):
        percent = round(percent + margin, 2)
        if progress == int(len(series) * percent):
            if (not tab):
                print(f"{ int(percent * 100) }% complete, {progress}/{len(series)}")
            else: 
                print(f"    { int(percent * 100) }% complete, {progress}/{len(series)}")
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


# the window and label series is the result of dataset deliver. Will automatically know whether the series has only channel 2 or has both channels.

# If only channel 2, then there will be four folder levels. The first level is the grand folder, this folder is named "exported csv files (only channel 2),"
# will be in the same directory as this python file, every exported csv files will be in this folder. 

# The second level is the name. Each folder in this level corresponds to one people, the name of the folder is simply the name, such as "19DA666D."
# Each people folder has three BAC level folders, the BAC level folder is the third folder level. The three BAC folder of each people is named "0," "5," and "10."

# In each BAC level folder, there's the exported window csvs and label csv. In python, for each video (say channel 2 of BAC level 0 of 19DA666D), the window is a 3D list,
# and the label is a 1D list. The first dimension of the window 3D list is the number of windows. 
# The number of windows is the number of exported csv files, each csv is a window. For label 1D list, then is just one csv file. 

# So in each BAC level folder, there are window_3D_list[0] number of window csv files and one label files. 
# The name of each window csv file is "CH02_{name}_{BAC level}_window_{window number}." For example, window 2389 of channel 2 of BAC level 5 of 4B725FCB will be in the folder
# "exported csv files (only channel 2)/4B725FCB/5," and is named "CH02_4B725FCB_5_window_2389.csv." For label file, the name is "CH02_{name}_{BAC level}_label."

# If include both channels, the grand folder's name is "exported csv files," and there will be five folder levels. The additional folder level is under BAC level, 
# each BAC level folder will has two channel folder,  each named "CH01" and "CH02." The window csv files and label csv file will be under the channel folder level. 
# So for window 2389 of channel 2 of BAC level 5 of 4B725FCB, the csv file will be under "exported csv files (only channel 2)/4B725FCB/5/CH02," 
# the name is still "CH02_4B725FCB_5_window_2389.csv" (if is channel 1 then the name is "CH01_4B725FCB_5_window_2389.csv"), everything else is the same.

# For each window csv file, there are seven attributes, "gaze_x," "gaze_y," "pose_Rx," "pose_Ry," "pose_Rz," "EAR," and "P-scale." 
# For each label csv file, there will be one attribute, "label"

# For one person with three BAC level and one channel, processing time is approximately four minutes, the csv files is about 1 GB
def dataset_export(window_series, label_series):
    # constructing name list, BAC list, channel list, and exported file path (will provide detail explanation few lines later)
    name_list = window_series.index.get_level_values(0).unique()
    BAC_list = window_series.index.get_level_values(1).unique()
    channel_list = []
    exported_file_path = ""

    # whether only contain channel 2 or not
    if (len(window_series.index.levels) == 2):
        only_channel_2 = True
    else:
        only_channel_2 = False
    
    if (only_channel_2):
        exported_file_path = "exported csv files (only channel 2)"
        channel_list = ["CH02"]
        print("has " + str(len(name_list)) + " people, each has " + str(len(BAC_list)) + " BAC levels, only one channel (channel 2), " +
              "total of " + str(len(name_list) * len(BAC_list)) + " videos,")
    else:
        exported_file_path = "exported csv files"
        channel_list = ["CH01", "CH02"]
        print("has " + str(len(name_list)) + " people, each has " + str(len(BAC_list)) + " BAC levels, each level has " + 
              str(len(channel_list)) + ", total of " + str(len(name_list) * len(BAC_list) * len(channel_list)) + " videos,")
    os.mkdir(exported_file_path)
    # so here, the name list will be the list of names (such as ["19DA666D", "4B725FCB"]), the BAC list is the list of BAC levels (
    # most liekly ["0", "5", "10"]), channel list will be ["CH02"] if only channel 2 and ["CH01", "CH02"] if both.
    # exported file path is the name of the grand folder 

    for i in range(len(name_list)):
        os.mkdir(exported_file_path + "/" + name_list[i])

        for j in range(len(BAC_list)):
            os.mkdir(exported_file_path + "/" + name_list[i] + "/" + BAC_list[j])

            if (only_channel_2):
                # export all windows of this BAC level of this name
                print("currently on person " + str(i) + " (" + name_list[i] + "), BAC level " +
                      BAC_list[j] + ", " + str(i * len(BAC_list) + j) + "/" + str(len(name_list) * len(BAC_list)))
                list_export(window_series[name_list[i]][BAC_list[j]], exported_file_path + "/" + name_list[i] + "/" + BAC_list[j] + 
                            "/CH02_" + name_list[i] + "_" + BAC_list[j])
                # export label list of this BAC level of this name
                label_dataframe = pd.DataFrame(label_series[name_list[i]][BAC_list[j]], columns = ["label"])
                label_dataframe.to_csv(exported_file_path + "/" + name_list[i] + "/" + BAC_list[j] + 
                                       "/CH02_" + name_list[i] + "_" + BAC_list[j] + "_label.csv", index = False) 
            
            else:
                for k in range(len(channel_list)):
                    # export all windows of this channel of this BAC level of this name
                    os.mkdir(exported_file_path + "/" + name_list[i] + "/" + BAC_list[j] + "/" + channel_list[k])
                    print("currently on person " + str(i) + " (" + name_list[i] + "), BAC level " +
                      BAC_list[j] + ", channel " + channel_list[k] + ", " + str(i * len(BAC_list) * len(channel_list) + j * len(channel_list) + k) +
                      "/" + str(len(name_list) * len(BAC_list) * len(channel_list)))
                    list_export(window_series[name_list[i]][BAC_list[j]][channel_list[k]], exported_file_path + "/" + name_list[i] + "/" + BAC_list[j] + "/" +
                            channel_list[k] + "/" + channel_list[k] + "_" + name_list[i] + "_" + BAC_list[j])
                    # export label list of this channel of this BAC level of this name
                    label_dataframe = pd.DataFrame(label_series[name_list[i]][BAC_list[j]][channel_list[k]], columns = ["label"])
                    label_dataframe.to_csv(exported_file_path + "/" + name_list[i] + "/" + BAC_list[j] + "/" +
                                           channel_list[k] + "/" + channel_list[k] + "_" + name_list[i] + "_" + BAC_list[j] + "_label.csv", index = False)
    
    print("everything finished! " + str(len(name_list) * len(BAC_list) * len(channel_list)) + "/" + str(len(name_list) * len(BAC_list) * len(channel_list)))
                
# the list is the 3D window list, file name should also include the path, the format is "{exported_file_path}/{name}/{BAC level}/CH02_{name}_{BAC level}" 
# for only channel 2 or "{exported_file_path}/{name}/{BAC level}/{channel}/{channel}_{name}_{BAC level}" for both channels.
# The function will export all windows 
def list_export(list, file_name):
    for i in range(len(list)):
        dataframe = pd.DataFrame(list[i], columns = ["gaze_x", "gaze_y", "EAR", "pose_Rx", "pose_Ry", "pose_Rz", "P-scale"])
        dataframe.to_csv(file_name + "_window_" + str(i) + ".csv", index = False)
        print_progress(list, 0.25, i + 1, True)


# %%
dataframe_series = overall_normalize("Step2Deliverables", ["19DA666D", "4B725FCB", "4B73E8AB", "4B73288B", "4BA9149B", "C60F509C", "C70E7E2C"], True)
# the entire name list is 
# ["19DA666D", "4B725FCB", "4B73E8AB", "4B73288B", "4BA9149B", "C60F509C", "C70E7E2C", "C719B90C", "C705493C", "DA9FFD3C", "DA96B45C", "DAA1F29C", "DAA49B9C", "DAA110DC"]


# %%
window_sereis, label_series  = dataset_deliver(dataframe_series)


# %%
dataset_export(window_sereis, label_series)

# %%


# %%

