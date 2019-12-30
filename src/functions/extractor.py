import os
import pandas as pd
import zipfile
import pydicom as dicom
import matplotlib.pyplot as plt
import cv2
import PIL
import csv
from tqdm import tqdm


class Unzipper():
    """Unzips the default zip file downloaded from Kaggle.
    """

    def __init__(self, path_to_zip_file, directory_to_extract_to):
        """Takes the path of the zip file and the directory
        to extract to. Either relative/absolute paths.

        Arguments:
            path_to_zip_file {str} -- path of zip file
            directory_to_extract_to {str} -- path to extract to
        """
        self.path_to_zip_file = path_to_zip_file
        self.directory_to_extract_to = directory_to_extract_to

    def unzip(self):
        """Unzips zip file to the specified directory.
        """
        print(f"Unzipping {self.path_to_zip_file} to "
              f"{self.directory_to_extract_to}")
        with zipfile.ZipFile(self.path_to_zip_file, 'r') as zip_ref:
            zip_ref.extractall(self.directory_to_extract_to)
        print("Unzipping finished")


class DCMConverter(Unzipper):
    """Converts DCM to PNG, unzipping if needed. Supply arguments
    as kwargs if you need to unzip files. Check init docstring.

    Arguments:
        Unzipper {class} -- Unzipper class
    """

    def __init__(
        self, dcm_path,
        output_path, output_csv_path,
        **kwargs
    ):
        """
        Converts DCM files to PNG files and stores information from DCM to csv.

        Arguments:
            dcm_path {str} -- The directory where the DCM files are stored.
            output_path {str} -- The directory where the output PNGs 
            will be stored.
            output_csv_path {str} -- Output directory + name of the 
            CSV containing info from DCM files.

        Keyword Arguments:
            path_to_zip_file {str} -- zipped folder location
            directory_to_extract_to {str} -- directory to unzip to

        """
        self.kwargs = kwargs
        self.dcm_path = dcm_path
        self.output_csv_path = output_csv_path
        self.output_path = output_path
        self.unzip_state = False
        self._check_unzipper()
        self.dicom_image_description = pd.read_csv(
            'https://raw.githubusercontent.com/vivek8981/DICOM-to-JP'
            'G/master/dicom_image_description.csv'
        )

    def convert_dcm(self):
        """Converts DCM to PNG, unzipping if needed.
        """
        if self.unzip_state:
            self.unzip()
        self._dcm_to_png()

    def _check_unzipper(self):
        """Internal function to check if files should
        be unzipped.
        """
        # Initialize Unzipper if kwargs argument given
        zip_vars = ['path_to_zip_file', 'directory_to_extract_to']
        if set(zip_vars).issubset(self.kwargs.keys()):
            super().__init__(
                self.kwargs.get('path_to_zip_file'),
                self.kwargs.get('directory_to_extract_to')
            )
        self.unzip_state = True

    def _dcm_to_png(self):
        """Converts DCM to PNG based on init parameters.
        """
        print("Starting to convert DCM to PNG")
        # Specify the .dcm folder path
        folder_path = self.dcm_path
        # Specify the .jpg/.png folder path
        jpg_folder_path = self.output_path
        images_path = os.listdir(folder_path)
        # list of attributes available in dicom image
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
        with open(self.output_csv_path, 'w', newline='') as csvfile:
            fieldnames = list(self.dicom_image_description["Description"])
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(fieldnames)
            for n, image in enumerate(tqdm(images_path)):
                ds = dicom.dcmread(os.path.join(folder_path, image))
                rows = []
                pixel_array_numpy = ds.pixel_array
                image = image.replace('.dcm', '.png')
                cv2.imwrite(
                    os.path.join(jpg_folder_path, image), pixel_array_numpy
                )
                for field in fieldnames:
                    if ds.data_element(field) is None:
                        rows.append('')
                    else:
                        x = str(ds.data_element(field)).replace("'", "")
                        y = x.find(":")
                        x = x[y+2:]
                        rows.append(x)
                writer.writerow(rows)
        print("Finished converting DCM to PNG")


class PatientInfoMerger():
    """Downloaded data from https://www.kaggle.com/c/rsna-pneumonia-detection
    -challenge/data contains a zip file which contains the main file with
    target labels and bounding boxes coordinates, as well as an additional
    file with extra target label information. This class merges these
    two files.
    """

    def __init__(self, main_path, supplement_path):
        """Initializes merger class with main file
        and extra target label file paths.

        Arguments:
            main_path {str} -- absolute path to main file
            supplement_path {str} -- absolute path to supplementary file
        """
        self.main_path = main_path
        self.supplement_path = supplement_path

    def merge_info(self):
        """Reads both files from init and merges them.

        Returns:
            pd.DataFrame -- Merged dataframe
        """
        labels = pd.read_csv(self.main_path)
        class_info = pd.read_csv(self.supplement_path)
        class_info.drop_duplicates(subset=['patientId', 'class'], inplace=True)
        labels.drop_duplicates(
            subset=[
                'patientId', 'x', 'y', 'width', 'height', 'Target'
            ], inplace=True
        )
        # Join both info
        return labels.merge(
            class_info,
            left_on='patientId',
            right_on='patientId',
            how='left'
        )
