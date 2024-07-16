import keypoint_moseq as kpms
import os 
import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd


def load_deeplabcut_csv(file_path):
    data = pd.read_csv(file_path, header=[0, 1, 2])
    return data

def get_bodyparts(data, scorer):
    return [col[0] for col in data[scorer].columns if col[0] != 'bodyparts']

def get_scorer(data):
    scorer = data.columns.get_level_values(0)[1]
    return scorer

def get_nosebase_arrays(xx, scorer):
    nose = 'Nose'
    base = 'TailBase'
    num_frames = len(xx)
    sample_frames = np.random.choice(range(int(num_frames/2), num_frames), size=1000, replace=False)
    nose_coords_x, nose_coords_y = np.array(xx[scorer][nose]['x'][sample_frames]), np.array(xx[scorer][nose]['y'][sample_frames])
    base_coords_x, base_coords_y = np.array(xx[scorer][base]['x'][sample_frames]), np.array(xx[scorer][base]['y'][sample_frames])
    
    return nose_coords_x, nose_coords_y, base_coords_x, base_coords_y

def mean_distance(nose_x, nose_y, tip_x, tip_y):
    """
    Calculate the mean distance between the tip and the nose across time.
    
    Parameters:
    - nose_x: numpy array of x coordinates of the nose
    - nose_y: numpy array of y coordinates of the nose
    - tip_x: numpy array of x coordinates of the tip
    - tip_y: numpy array of y coordinates of the tip
    
    Returns:
    - mean_dist: mean distance between the tip and the nose
    """
    # Calculate the Euclidean distance at each time point
    distances = np.sqrt((tip_x - nose_x)**2 + (tip_y - nose_y)**2)
    
    # Compute the mean distance
    mean_dist = np.mean(distances)
    
    return mean_dist

# Transform coordinates
def transform_coordinates(points, H):
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    transformed_points_homogeneous = np.dot(H, points_homogeneous.T).T
    transformed_points = transformed_points_homogeneous[:, :2] / transformed_points_homogeneous[:, 2, np.newaxis]
    return transformed_points

# Extract body parts correctly
def extract_and_transform_coordinates(data, scorer, bodypart, H):

    x_coords = data[scorer][(bodypart, 'x')].values.copy()

    y_coords = data[scorer][(bodypart, 'y')].values.copy()

    points = np.vstack((x_coords, y_coords)).T
 
    transformed_points = transform_coordinates(points, H)
    
    return transformed_points, points
    
# Save transformed coordinates back to CSV

def save_transformed_coordinates_with_headers(data, scorer, transformed_coords_dict, output_file):
    original_data = data.copy()
    for bodypart in transformed_coords_dict.keys():
        # print(original_data.loc[:,(scorer, bodypart, 'x')]==transformed_coords_dict[bodypart][:, 0])
        original_data.loc[:,(scorer, bodypart, 'x')] = transformed_coords_dict[bodypart][:, 0]
        # print(original_data.loc[:,(scorer, bodypart, 'x')]==transformed_coords_dict[bodypart][:, 0])
        original_data.loc[:,(scorer, bodypart, 'y')] = transformed_coords_dict[bodypart][:, 1]
        # original_data[scorer][bodypart]['x'] = transformed_coords_dict[bodypart][:, 0]
        # original_data[scorer][bodypart]['y'] = transformed_coords_dict[bodypart][:, 1]
    original_data.to_csv(output_file, index=False)

def read_multi_level_header(source_csv_path, num_header_rows):
    """
    Reads the multi-level header from the source CSV file.
    
    Parameters:
    source_csv_path (str): Path to the source CSV file.
    num_header_rows (int): Number of header rows in the CSV file.
    
    Returns:
    list of lists: A list containing the header rows.
    """
    header_rows = []
    with open(source_csv_path, mode='r', newline='') as source_csv:
        reader = csv.reader(source_csv)
        for _ in range(num_header_rows):
            header_rows.append(next(reader))
    return header_rows

def add_multi_level_header_to_target(header_rows, target_csv_path):
    """
    Adds the multi-level header to the target CSV file.
    
    Parameters:
    header_rows (list of lists): A list containing the header rows.
    target_csv_path (str): Path to the target CSV file.
    """
    with open(target_csv_path, mode='a', newline='') as target_csv:
        writer = csv.writer(target_csv)
        writer.writerows(header_rows)

def transfer_multi_level_header(source_csv_path, target_csv_path, num_header_rows):
    """
    Transfers the multi-level header from the source CSV file to the target CSV file.
    
    Parameters:
    source_csv_path (str): Path to the source CSV file.
    target_csv_path (str): Path to the target CSV file.
    num_header_rows (int): Number of header rows in the CSV file.
    """
    header_rows = read_multi_level_header(source_csv_path, num_header_rows)
    add_multi_level_header_to_target(header_rows, target_csv_path)



# Example usage

if __name__ == '__main__':
    videos_dir = '/Users/AdamHarris/Desktop/parkinsons_files/videos'
    keypoints_dir = '/Users/AdamHarris/Desktop/parkinsons_files/keypoints'
    keypoint_data_path = keypoints_dir #"/Users/AdamHarris/Desktop/correct_filenames/keypoints"
    keypoints_csvs = os.listdir(keypoints_dir)
    keypoints_paths = []

    for i in keypoints_csvs:
        keypoints_paths.append(os.path.join(keypoint_data_path, i))
        
    file_paths = keypoints_paths # Add more paths as needed
    
    output_folder = "/Users/AdamHarris/Desktop/parkinsons_files/keypoints_transformed"
    output_files =  [os.path.join(output_folder, i) for i in keypoints_csvs]
    
    loaded_points = np.load("/Users/AdamHarris/Desktop/corner_points.npy", allow_pickle=True).item()
    print("Loaded corner points:", loaded_points)

    reference_points = np.array([[50,300],
    [50,50],
    [600,50],
    [600,300]])

    camera_points = {}

    for i, j in enumerate(loaded_points):
        camera_points[f"c{i+2}"]=loaded_points[j]
        homographies = {}

    for camera in camera_points:
        homographies[camera] = compute_homography(camera_points[camera], reference_points)
        print(camera_points[camera])
        test_points = np.array([[200,200],
                    [100,100],
                    [150,150],
                    [123,123]])
        print(homographies[camera])
        print(transform_coordinates(test_points, homographies[camera]))

        homographies_list = [homographies[i] for i in camera_each_video]

    xx = 0
    for file_path, H, output_file, cc in zip(file_paths, homographies_list, output_files, camera_each_video):

        data = load_deeplabcut_csv(file_path)
        scorer = data.columns.get_level_values(0)[1]
        print(file_path)
        # get list of body parts and scorer

        bodyparts = list(data.columns.get_level_values(1).unique())
        bodyparts = [i for i in bodyparts if i != 'bodyparts']
        
        transformed_coords_dict = {}
        for bodypart in bodyparts:
            transformed_coords, points_ = extract_and_transform_coordinates(data, scorer, bodypart, H)
            if bodypart=='Nose':
                plt.scatter(transformed_coords[:,0], transformed_coords[:,1], c='red')
                plt.scatter(points_[:,0], points_[:,1], c='black')
                plt.title(cc)
                plt.show()

            transformed_coords_dict[bodypart] = transformed_coords
        
        save_transformed_coordinates_with_headers(data,scorer, transformed_coords_dict, output_file) #bodyparts
        print(f"Transformed coordinates saved to {output_file}")




    num_header_rows =4  # Adjust this based on the actual number of header rows in your CSV

    for csv_in, csv_out in zip(keypoints_paths, output_files):
        transfer_multi_level_header(csv_in, csv_out, num_header_rows)

