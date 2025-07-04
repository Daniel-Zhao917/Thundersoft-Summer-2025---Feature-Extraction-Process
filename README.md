# Thundersoft-Summer-2025---Feature-Extraction-Process
## Extraction
So in our ThunderSoft databset at https://thundersoft.feishu.cn/drive/folder/EC24fgOnFlGNE3dnE0Fc8PNgnAh, we have two people, named "4B725FCB" and "19DA666D". Each people has three BAC levels. After extracting, we have 12 csv files, this is because for each BAC level of a people, we have two chapters, so the structure is 2 (the two people) × 3 (BAC levels) × 2 (chapters). The number of attributes for each csv file is not the same, but each all have "gaze_angle_x," "gaze_angle_y," "pose_Rx," "pose_Ry," "pose_Rz," "EAR," "Pupil_Scale (Left)," and "Pupil_Scale (Right)." So there's no "P-scale," but has "Pupil_Scale (Left)" and "Pupil_Scale (Right)."

## Overall Data Processing Structure
There are 4 steps that process the 12 raw csv data. The first step is to calculate the P-scale and normalize the 7 attributes ("gaze_angle_x," "gaze_angle_y," "pose_Rx," "pose_Ry," "pose_Rz," "EAR," and "P-scale"). The second step is to extract these 7 attributes. The third step is to use the sliding window algorithm to create the dataset. The fourth step is to create the label for each dataset. We will explain each of the 4 steps in a section

## Normalization
The calculation of P-scale is simply the average of "Pupil_Scale (Left)" and "Pupil_Scale (Right)." The normalization is z-score normalizing, (value - mean) / std.

## Extraction
We simply just extract "gaze_angle_x," "gaze_angle_y," "pose_Rx," "pose_Ry," "pose_Rz," "EAR," and "P-scale." Please note that we didn't change the name of "gaze_angle_x," "gaze_angle_y" to "gaze_x" and "gaze_y."

## Sliding Window
Take the 0 BAC level, chapter 1, person "4B725FCB" as an example. This csv file has 12423 rows. For window 1, we create a Python built-in list (so not a PyTorch tensor) with the dimension of 150 × 7. This window consist of frame 1 - 150 with all the 7 attributes. Let's say this list is called "list_1". list_1[0] will return a list with 7 elements, each is an attribute. Thus, this list is row-oriented, not the pandas style column-oriented (if is column-oriented, list_1[0] will return the first column (in this case, "gaze_angle_x", with a list of 150 elements)). For window 2, is frame 2 - 151. Because we have 12423 rows, we have a total of 12274 windows, the last one consists of frame 12274 to 12423. In the end, we have a list with dimension of (frame number - 149) × 150 × 7. list[0] will return the first window (a 150 × 7 list), list[0] [0] will return a list of 7. Because there are 12 csv files, we have 12 3D lists, each is a window.

## Label
For each raw csv file, we create a 1D list with the length of frame number - 149. So for the above example, we have a list of 12274 elements, all of which are 0 (so BAC level 0.00). Because there are 12 csv files, we have 12 1D lists, each is a label. Please note that label is separate from window, so we actually have 24 lists, 12 are windows, 12 are labels.
